"""Tests der ComfyUI-Anbindung gegen einen simulierten ComfyUI-Server."""

import io
import json
import threading

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from bildwerk.ai import ProviderError, ProviderRegistry
from bildwerk.ai import comfyui_provider as comfy
from bildwerk.ai.base import GenerateOptions
from bildwerk.ai.comfy_workflows import WorkflowError, WorkflowLibrary, analyze, fill
from bildwerk.ai.comfyui_provider import ComfyUIProvider
from bildwerk.ai.pipeline import ai_edit
from bildwerk.selection import RectShape, Selection
from bildwerk.server import create_app

from .conftest import encode, noise_image

# Ein typischer, aus ComfyUI exportierter Workflow (API-Format) im Stil von Z-Image Turbo
ZIMAGE = {
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "z-image", "images": ["43", 0]}},
    "39": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2"}},
    "40": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
    "41": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "42": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["45", 0]}},
    "43": {"class_type": "VAEDecode", "inputs": {"samples": ["44", 0], "vae": ["40", 0]}},
    "44": {
        "class_type": "KSampler",
        "inputs": {"seed": 5, "steps": 8, "cfg": 1, "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1,
                   "model": ["46", 0], "positive": ["45", 0], "negative": ["42", 0], "latent_image": ["41", 0]},
    },
    "45": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["39", 0]}, "_meta": {"title": "Prompt"}},
    "46": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo.safetensors", "weight_dtype": "default"}},
}

# FLUX-artig mit SamplerCustomAdvanced, RandomNoise und Guider
FLUX = {
    "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "x", "clip": ["9", 0]}},
    "2": {"class_type": "FluxGuidance", "inputs": {"guidance": 3.5, "conditioning": ["1", 0]}},
    "3": {"class_type": "BasicGuider", "inputs": {"model": ["8", 0], "conditioning": ["2", 0]}},
    "4": {"class_type": "RandomNoise", "inputs": {"noise_seed": 1}},
    "5": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "6": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["4", 0], "guider": ["3", 0], "sampler": ["10", 0], "sigmas": ["11", 0], "latent_image": ["5", 0]}},
    "7": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": "flux"}},
    "8": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux.safetensors", "weight_dtype": "default"}},
    "9": {"class_type": "CLIPLoader", "inputs": {"clip_name": "t5.safetensors", "type": "flux"}},
    "12": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["13", 0]}},
}

EDIT_WF = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "beispiel.png"}},
    "2": {"class_type": "TextEncodeQwenImageEdit", "inputs": {"prompt": "{{prompt}}", "clip": ["5", 0], "image": ["1", 0]}},
    "3": {"class_type": "KSampler", "inputs": {"seed": 0, "positive": ["2", 0], "negative": ["2", 0], "latent_image": ["4", 0]}},
    "6": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "edit"}},
}


@pytest.fixture(autouse=True)
def clean_state():
    for state in (comfy._probe_cache, comfy._active, comfy._cancelled):
        state.clear()
    yield
    for state in (comfy._probe_cache, comfy._active, comfy._cancelled):
        state.clear()


class FakeComfy:
    """Simuliert die ComfyUI-Endpunkte, die Bildwerk nutzt."""

    def __init__(self, checkpoints=("sdxl_base.safetensors",), pending_polls=1, history=None, prompt_error=None):
        self.checkpoints = list(checkpoints)
        self.pending_polls = pending_polls
        self.history_override = history
        self.prompt_error = prompt_error
        self.graphs = []
        self.uploads = []
        self.requests = []
        self.output_size = (64, 48)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append((request.method, request.url.path))
        path = request.url.path
        if path == "/system_stats":
            return httpx.Response(200, json={"system": {"os": "nt"}, "devices": [{"name": "cuda:0 NVIDIA GeForce RTX 5060 Laptop GPU : cudaMallocAsync", "type": "cuda", "vram_total": 8 * 1024**3}]})
        if path == "/models/checkpoints":
            return httpx.Response(200, json=self.checkpoints)
        if path == "/prompt":
            if self.prompt_error:
                return httpx.Response(400, json=self.prompt_error)
            self.graphs.append(json.loads(request.content)["prompt"])
            return httpx.Response(200, json={"prompt_id": f"p{len(self.graphs)}", "number": 1, "node_errors": {}})
        if path.startswith("/history/"):
            prompt_id = path.rsplit("/", 1)[-1]
            if self.pending_polls > 0:
                self.pending_polls -= 1
                return httpx.Response(200, json={})
            if self.history_override is not None:
                return httpx.Response(200, json={prompt_id: self.history_override})
            # so viele Bilder, wie der Workflow im Stapel (batch_size) erzeugt
            graph = self.graphs[int(prompt_id[1:]) - 1]
            batch = max([n["inputs"].get("batch_size", 1) for n in graph.values()] or [1])
            images = [{"filename": f"{prompt_id}_{i:05}_.png", "subfolder": "", "type": "output"} for i in range(batch)]
            outputs = {"9": {"images": images}}
            return httpx.Response(200, json={prompt_id: {"outputs": outputs, "status": {"status_str": "success", "completed": True, "messages": []}}})
        if path == "/view":
            return httpx.Response(200, content=encode(Image.new("RGB", self.output_size, (10, 200, 30))))
        if path == "/upload/image":
            boundary = request.headers["content-type"].split("boundary=")[1].encode()
            for part in request.content.split(b"--" + boundary):
                if b'name="image"' in part:
                    self.uploads.append(Image.open(io.BytesIO(part.split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n"))))
            return httpx.Response(200, json={"name": "bildwerk_upload.png", "subfolder": "", "type": "input"})
        if path in ("/interrupt", "/queue"):
            return httpx.Response(200)
        return httpx.Response(404)


def make_registry(settings, fake):
    client = httpx.Client(transport=httpx.MockTransport(fake))
    settings.set("comfyui_url", "http://comfy.test:8188")
    return ProviderRegistry(settings, client=client, classes=(ComfyUIProvider,))


# -- Workflows auswerten ------------------------------------------------------------------------
def test_analyze_zimage_style_workflow():
    info = analyze(ZIMAGE)
    assert info.kind == "generate"
    assert info.prompt == ("45", "text")
    assert info.negative is None  # Negativ ist ein ConditioningZeroOut ohne Text
    assert info.size_nodes == ["41"] and info.seeds == [("44", "seed")] and info.batch == [("41", "batch_size")]
    graph = fill(ZIMAGE, info, prompt="ein Leuchtturm", width=1216, height=832, seed=77, count=2)
    assert graph["45"]["inputs"]["text"] == "ein Leuchtturm"
    assert graph["41"]["inputs"] == {"width": 1216, "height": 832, "batch_size": 2}
    assert graph["44"]["inputs"]["seed"] == 77
    assert ZIMAGE["45"]["inputs"]["text"] == "a cat"  # Original bleibt unverändert


def test_analyze_flux_style_and_placeholders():
    info = analyze(FLUX)
    assert info.prompt == ("1", "text") and info.seeds == [("4", "noise_seed")]
    edit = analyze(EDIT_WF)
    assert edit.kind == "edit" and edit.image_node == "1" and edit.prompt == ("2", "prompt")
    graph = fill(EDIT_WF, edit, prompt="mehr Sonne", image_name="neu.png")
    assert graph["2"]["inputs"]["prompt"] == "mehr Sonne" and graph["1"]["inputs"]["image"] == "neu.png"
    assert any("Eingangsbild" in line for line in edit.summary(EDIT_WF))


@pytest.mark.parametrize(
    "graph,message",
    [
        ({"nodes": [], "links": []}, "Export \\(API\\)"),
        ({"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "x"}}}, "Save Image"),
        ({"1": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}}}, "Textfeld"),
        ([], "keinen ComfyUI-Workflow"),
    ],
)
def test_unusable_workflows(graph, message):
    with pytest.raises(WorkflowError, match=message):
        analyze(graph)


def test_workflow_library(tmp_path):
    library = WorkflowLibrary(tmp_path / "comfyui")
    stored = library.add("Z-Image Turbo", ZIMAGE)
    assert stored.kind == "generate" and library.get(stored.id).name == "Z-Image Turbo"
    assert [w.id for w in library.list()] == [stored.id]
    library.remove(stored.id)
    assert library.list() == []
    with pytest.raises(WorkflowError):
        library.remove("../boese")


# -- Anbieter -------------------------------------------------------------------------------------
def test_availability_lists_models(settings):
    fake = FakeComfy()
    registry = make_registry(settings, fake)
    WorkflowLibrary(settings.config_dir / "comfyui").add("Z-Image", ZIMAGE)
    info = registry.describe()[0]
    assert info["configured"] and "RTX 5060" in info["status"] and "8 GB" in info["status"]
    ids = [m["id"] for m in info["models"]]
    assert ids[0].startswith("wf:z-image") and "ckpt:sdxl_base.safetensors" in ids
    assert info["default_model"] == ids[0] and info["model_scope"] == "card"
    assert info["enhance"] is False


def test_unreachable_server(settings):
    def offline(request):
        raise httpx.ConnectError("verbindung abgelehnt")

    settings.set("comfyui_url", "http://127.0.0.1:9")
    registry = ProviderRegistry(settings, client=httpx.Client(transport=httpx.MockTransport(offline)), classes=(ComfyUIProvider,))
    info = registry.describe()[0]
    assert not info["configured"] and "Nicht erreichbar" in info["status"]
    with pytest.raises(ProviderError, match="Nicht erreichbar"):
        registry.get("comfyui")


def test_generate_with_checkpoint(settings):
    fake = FakeComfy(pending_polls=2)
    provider = make_registry(settings, fake).get("comfyui", "generate")
    images = provider.generate(GenerateOptions(prompt="Berge", model="ckpt:sdxl_base.safetensors", aspect_ratio="16:9",
                                               negative_prompt="Text", seed=3, count=1, extra={"steps": 30, "cfg": 7}))
    assert len(images) == 1 and images[0].size == (64, 48)
    graph = fake.graphs[0]
    assert graph["1"]["inputs"]["ckpt_name"] == "sdxl_base.safetensors"
    assert graph["4"]["inputs"] == {"width": 1344, "height": 768, "batch_size": 1}
    assert graph["5"]["inputs"]["steps"] == 30 and graph["5"]["inputs"]["seed"] == 3
    assert graph["3"]["inputs"]["text"] == "Text"


def test_generate_with_workflow_uses_batch(settings):
    fake = FakeComfy(pending_polls=0)
    registry = make_registry(settings, fake)
    stored = WorkflowLibrary(settings.config_dir / "comfyui").add("Flux", FLUX)
    images = registry.get("comfyui").generate(GenerateOptions(prompt="Wald", model=f"wf:{stored.id}", count=3, seed=10))
    assert len(images) == 3 and len(fake.graphs) == 1
    assert fake.graphs[0]["5"]["inputs"]["batch_size"] == 3 and fake.graphs[0]["4"]["inputs"]["noise_seed"] == 10
    assert fake.graphs[0]["1"]["inputs"]["text"] == "Wald"


def test_generate_with_workflow_loops_without_batch(settings):
    fake = FakeComfy(pending_polls=0)
    registry = make_registry(settings, fake)
    graph = json.loads(json.dumps(FLUX))
    del graph["5"]["inputs"]["batch_size"]
    stored = WorkflowLibrary(settings.config_dir / "comfyui").add("Flux ohne Stapel", graph)
    images = registry.get("comfyui").generate(GenerateOptions(prompt="Wald", model=f"wf:{stored.id}", count=2, seed=10))
    assert len(images) == 2 and len(fake.graphs) == 2
    assert [g["4"]["inputs"]["noise_seed"] for g in fake.graphs] == [10, 11]


def test_inpaint_with_checkpoint_uses_alpha_mask(settings):
    fake = FakeComfy(pending_polls=0)
    fake.output_size = (1024, 1024)
    provider = make_registry(settings, fake).get("comfyui", "edit")
    image = noise_image(300, 300)
    sel = Selection(shapes=[RectShape(x=100, y=100, w=100, h=100)])
    change = ai_edit(provider, image, sel, "Blumen", "ckpt:sdxl_base.safetensors", options={"denoise": 0.8, "steps": 20})
    assert change.kind == "patch" and change.box == (100, 100, 200, 200)
    upload = fake.uploads[0]
    assert upload.mode == "RGBA" and upload.size == (1024, 1024)
    alpha = np.asarray(upload.getchannel("A"))
    assert alpha[512, 512] == 0 and alpha[5, 5] == 255  # Auswahl = durchsichtig
    graph = fake.graphs[0]
    assert graph["4"] == {"class_type": "LoadImage", "inputs": {"image": "bildwerk_upload.png"}}
    assert graph["5"]["class_type"] == "InpaintModelConditioning"
    assert graph["6"]["inputs"]["denoise"] == 0.8 and graph["6"]["inputs"]["steps"] == 20


def test_wrong_workflow_kind(settings):
    fake = FakeComfy()
    registry = make_registry(settings, fake)
    stored = WorkflowLibrary(settings.config_dir / "comfyui").add("Nur Generieren", ZIMAGE)
    with pytest.raises(ProviderError, match="nicht zum Bearbeiten"):
        registry.get("comfyui").edit(Image.new("RGB", (64, 64)), Image.new("L", (64, 64), 255), "x", f"wf:{stored.id}")


def test_errors_are_explained(settings):
    error = {
        "error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"},
        "node_errors": {"1": {"class_type": "CheckpointLoaderSimple", "errors": [{"message": "Value not in list", "details": "ckpt_name: 'fehlt.safetensors' not in []"}]}},
    }
    provider = make_registry(settings, FakeComfy(prompt_error=error)).get("comfyui")
    with pytest.raises(ProviderError, match="fehlt.safetensors"):
        provider.generate(GenerateOptions(prompt="x", model="ckpt:fehlt.safetensors"))

    oom = {"outputs": {}, "status": {"status_str": "error", "completed": False, "messages": [
        ["execution_error", {"node_type": "KSampler", "exception_message": "CUDA out of memory. Tried to allocate 2 GiB"}]]}}
    provider = make_registry(settings, FakeComfy(pending_polls=0, history=oom)).get("comfyui")
    with pytest.raises(ProviderError, match="Grafikspeicher"):
        provider.generate(GenerateOptions(prompt="x", model="ckpt:sdxl_base.safetensors"))


def test_cancel_stops_only_the_given_job(settings):
    fake = FakeComfy(pending_polls=10_000)
    registry = make_registry(settings, fake)
    provider = registry.get("comfyui")
    results = {}

    def run(job):
        try:
            provider.generate(GenerateOptions(prompt="x", model="ckpt:sdxl_base.safetensors", extra={"job_id": job}))
            results[job] = "fertig"
        except ProviderError as exc:
            results[job] = str(exc)

    workers = {job: threading.Thread(target=run, args=(job,)) for job in ("a", "b")}
    for worker in workers.values():
        worker.start()
    for _ in range(300):
        if len(comfy._active) == 2:
            break
        threading.Event().wait(0.01)
    assert comfy.cancel(registry.client, "a") == 1
    workers["a"].join(timeout=5)
    assert results == {"a": "Abgebrochen."}
    assert ("POST", "/interrupt") in fake.requests and ("POST", "/queue") in fake.requests
    fake.pending_polls = 0  # Auftrag b darf fertig werden
    workers["b"].join(timeout=5)
    assert results["b"] == "fertig"

    # Abbruch, bevor der Auftrag überhaupt bei ComfyUI angekommen ist
    queued = len(fake.graphs)
    assert comfy.cancel(registry.client, "c") == 0
    with pytest.raises(ProviderError, match="Abgebrochen"):
        provider.generate(GenerateOptions(prompt="x", model="ckpt:sdxl_base.safetensors", extra={"job_id": "c"}))
    assert len(fake.graphs) == queued


# -- Server-Endpunkte ------------------------------------------------------------------------------
def test_comfyui_endpoints(settings):
    fake = FakeComfy()
    registry = make_registry(settings, fake)
    client = TestClient(create_app(settings=settings, registry=registry, allowed_hosts=["testserver"]))
    status = client.get("/api/comfyui").json()
    assert status["reachable"] and status["checkpoints"] == ["sdxl_base.safetensors"]
    upload = client.post("/api/comfyui/workflows", files={"file": ("z-image.json", json.dumps(ZIMAGE), "application/json")}, data={"name": "Z-Image"})
    assert upload.status_code == 200 and upload.json()["workflow"]["kind"] == "generate"
    workflow_id = upload.json()["workflow"]["id"]
    bad = client.post("/api/comfyui/workflows", files={"file": ("ui.json", json.dumps({"nodes": [], "links": []}), "application/json")})
    assert bad.status_code == 400 and "API" in bad.json()["detail"]
    assert client.delete(f"/api/comfyui/workflows/{workflow_id}").json()["workflows"] == []
    changed = client.put("/api/comfyui/url", json={"url": "http://192.168.0.5:8188/"}).json()
    assert changed["configured_url"] == "http://192.168.0.5:8188"
    assert client.put("/api/comfyui/url", json={"url": "kein url"}).status_code == 422
    generated = client.post("/api/ai/generate", json={"provider": "comfyui", "prompt": "Ballon", "model": "ckpt:sdxl_base.safetensors", "options": {"steps": 12}})
    assert generated.status_code == 200 and fake.graphs[-1]["5"]["inputs"]["steps"] == 12
    assert client.post("/api/ai/cancel").json() == {"cancelled": 0}
    assert client.post("/api/ai/cancel", json={"job_id": "auftrag-1"}).json() == {"cancelled": 0}
    assert client.post("/api/ai/cancel", json={"job_id": "kein gültiger"}).status_code == 422
