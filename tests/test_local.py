"""Tests der lokalen KI (Real-ESRGAN über ONNX Runtime).

Die Testdateien in ``tests/fixtures`` wurden mit echtem PyTorch und den Original-Architekturen
aus Real-ESRGAN erzeugt (``torch.save`` bzw. Ein-/Ausgaben der Netze).
"""

import io
import pickle
import time
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

import onnxruntime as ort  # noqa: E402

from bildwerk.ai import ProviderError, ProviderRegistry  # noqa: E402
from bildwerk.ai.local import esrgan, runtime  # noqa: E402
from bildwerk.ai.local.models import LocalModel, ModelError, ModelStore, WeightFile  # noqa: E402
from bildwerk.ai.local.torchfile import WeightFileError, load_state_dict  # noqa: E402
from bildwerk.ai.local_provider import LocalProvider  # noqa: E402
from bildwerk.ai.pipeline import ai_enhance  # noqa: E402
from bildwerk.selection import RectShape, Selection  # noqa: E402
from bildwerk.server import create_app  # noqa: E402

from .conftest import noise_image  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def session_for(model_bytes):
    return ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])


@pytest.mark.parametrize(
    "name,builder",
    [("tiny_srvgg", esrgan.srvgg_onnx), ("tiny_rrdb", esrgan.rrdbnet_onnx)],
)
def test_onnx_graph_matches_pytorch_reference(name, builder):
    state = load_state_dict(FIXTURES / f"{name}.pth")
    io_data = np.load(FIXTURES / f"{name}_io.npz")
    out = session_for(builder(state)).run(None, {"input": io_data["input"]})[0]
    assert out.shape == io_data["output"].shape
    np.testing.assert_allclose(out, io_data["output"], atol=1e-5)


def test_state_dict_loading():
    state = load_state_dict(FIXTURES / "tiny_srvgg.pth")
    assert state["body.0.weight"].shape == (4, 3, 3, 3)
    assert state["body.1.weight"].shape == (4,)  # PReLU
    assert all(v.dtype == np.float32 for v in state.values())
    rrdb = load_state_dict((FIXTURES / "tiny_rrdb.pth").read_bytes())
    assert "body.0.rdb3.conv5.weight" in rrdb


def test_rejects_invalid_and_malicious_files():
    with pytest.raises(WeightFileError, match="ZIP"):
        load_state_dict(b"kein zip")

    class Evil:
        def __reduce__(self):
            import os

            return (os.system, ("echo boese",))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps(Evil(), protocol=2))
    with pytest.raises(WeightFileError, match="Unerwarteter Inhalt"):
        load_state_dict(buf.getvalue())


def test_wrong_weights_give_clear_error():
    state = load_state_dict(FIXTURES / "tiny_srvgg.pth")
    with pytest.raises(WeightFileError):
        esrgan.rrdbnet_onnx(state)
    broken = dict(state)
    del broken["body.1.weight"]
    with pytest.raises(WeightFileError, match="fehlt"):
        esrgan.srvgg_onnx(broken)


def test_blending_mixes_weights():
    a = load_state_dict(FIXTURES / "tiny_srvgg.pth")
    b = load_state_dict(FIXTURES / "tiny_srvgg_b.pth")
    mixed = esrgan.blend_states([a, b], (0.5, 0.5))
    np.testing.assert_allclose(mixed["body.0.weight"], (a["body.0.weight"] + b["body.0.weight"]) / 2, rtol=1e-6)


@pytest.mark.parametrize("size", [(1, 1), (5, 3), (40, 23), (64, 64)])
def test_tiled_upscaling_matches_single_pass(size):
    session = session_for(esrgan.srvgg_onnx(load_state_dict(FIXTURES / "tiny_srvgg.pth")))
    rng = np.random.default_rng(0)
    image = rng.random((size[1], size[0], 3), dtype=np.float32)
    whole = runtime.upscale_tiled(session, image, tile=256, pad=16)
    tiled = runtime.upscale_tiled(session, image, tile=8, pad=8)
    assert whole.shape == (size[1] * 4, size[0] * 4, 3)
    np.testing.assert_allclose(tiled, whole, atol=1e-5)


# -- Modellverwaltung mit simuliertem Download ----------------------------------------------
def tiny_model(blend=False):
    files = (WeightFile("a.pth", "https://example.invalid/a.pth", 0.01),)
    if blend:
        files += (WeightFile("b.pth", "https://example.invalid/b.pth", 0.01),)
    return LocalModel("tiny", "Mini", "Testmodell", files, "srvgg", blend=(0.5, 0.5) if blend else None, tile=32, pad=8)


def fake_downloads(missing=()):
    calls = []

    def respond(request):
        name = request.url.path.rsplit("/", 1)[-1]
        calls.append(name)
        if name in missing:
            return httpx.Response(404)
        source = {"a.pth": "tiny_srvgg.pth", "b.pth": "tiny_srvgg_b.pth"}[name]
        return httpx.Response(200, content=(FIXTURES / source).read_bytes())

    return calls, httpx.Client(transport=httpx.MockTransport(respond))


def test_store_downloads_converts_and_upscales(tmp_path):
    calls, client = fake_downloads()
    store = ModelStore(tmp_path / "models", client)
    model = tiny_model(blend=True)
    assert not store.installed(model)
    result = store.upscale(model, noise_image(20, 12))
    assert result.size == (80, 48) and result.mode == "RGB"
    assert sorted(calls) == ["a.pth", "b.pth"] and store.installed(model)
    store.upscale(model, noise_image(8, 8))
    assert len(calls) == 2  # nicht erneut geladen
    assert store.device() == "Prozessor (CPU)"


def test_store_reports_download_errors(tmp_path):
    _, client = fake_downloads(missing=("a.pth",))
    store = ModelStore(tmp_path / "models", client)
    with pytest.raises(ModelError, match="HTTP 404"):
        store.ensure(tiny_model())
    assert not (tmp_path / "models" / "a.pth.part").exists()


def test_corrupt_file_is_reported(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "a.pth").write_bytes(b"defekt")
    store = ModelStore(tmp_path / "models", httpx.Client())
    with pytest.raises(ModelError, match="löschen"):
        store.session(tiny_model())


def test_local_provider_via_registry(settings, monkeypatch):
    import bildwerk.ai.local_provider as provider_module

    model = tiny_model()
    monkeypatch.setattr(provider_module, "CATALOG", {"tiny": model})
    _, client = fake_downloads()
    registry = ProviderRegistry(settings, client=client, classes=(LocalProvider,))
    info = registry.describe()[0]
    assert info["configured"] and info["local"] and not info["requires_key"]
    assert info["capabilities"] == ["upscale"] and info["enhance"]

    provider = registry.get("local", "upscale")
    image = noise_image(200, 120)
    sel = Selection(shapes=[RectShape(x=50, y=30, w=60, h=40)])
    change = ai_enhance(provider, image, sel, None, "tiny")
    assert change.kind == "patch" and change.box == (50, 30, 110, 70)
    zoom = ai_enhance(provider, image, sel, None, "tiny", as_new=True)
    assert zoom.image.size == (240, 160)
    with pytest.raises(ProviderError, match="kann keine Bilder generieren"):
        registry.get("local", "generate")


def test_local_provider_without_onnxruntime(settings, monkeypatch):
    import bildwerk.ai.local_provider as provider_module

    monkeypatch.setattr(provider_module, "onnxruntime_available", lambda: False)
    registry = ProviderRegistry(settings, classes=(LocalProvider,))
    info = registry.describe()[0]
    assert not info["configured"] and "onnxruntime" in info["status"]
    with pytest.raises(ProviderError, match="onnxruntime"):
        registry.get("local")


def test_local_endpoints(settings, monkeypatch):
    import bildwerk.ai.local.models as models_module
    import bildwerk.server as server_module

    model = tiny_model()
    monkeypatch.setattr(models_module, "CATALOG", {"tiny": model})
    monkeypatch.setattr(server_module, "LOCAL_MODELS", {"tiny": model})
    calls, client = fake_downloads()
    registry = ProviderRegistry(settings, client=client, classes=(LocalProvider,))
    with TestClient(create_app(settings=settings, registry=registry, allowed_hosts=["testserver"])) as http:
        status = http.get("/api/local").json()
        assert status["available"] and status["device"] == "Prozessor (CPU)"
        assert status["models"][0]["id"] == "tiny" and not status["models"][0]["installed"]
        assert http.post("/api/local/models/unbekannt/download").status_code == 404
        assert http.post("/api/local/models/tiny/download").status_code == 200
        for _ in range(250):
            entry = http.get("/api/local").json()["models"][0]
            if entry["installed"] and entry["progress"] is None:
                break
            time.sleep(0.02)
        assert entry["installed"] and entry["error"] is None and calls == ["a.pth"]
        providers = http.get("/api/ai/providers").json()["providers"]
        assert providers[0]["id"] == "local" and providers[0]["configured"]
