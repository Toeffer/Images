"""Tests der KI-Anbieter mit simulierten HTTP-Antworten (es werden keine echten APIs aufgerufen)."""

import base64
import io
import json

import httpx
import numpy as np
import pytest
from PIL import Image

from bildwerk.ai import ProviderError, ProviderRegistry
from bildwerk.ai.base import GenerateOptions
from bildwerk.ai.gemini_provider import GeminiProvider
from bildwerk.ai.openai_provider import OpenAIProvider
from bildwerk.ai.pipeline import ENHANCE_PROMPT, ai_edit, ai_enhance, fit_aspect
from bildwerk.ai.stability_provider import StabilityProvider
from bildwerk.engine import NewImage
from bildwerk.selection import RectShape, Selection
from bildwerk.settings import Settings

from .conftest import FakeProvider, FakeUpscaler, encode, noise_image


def b64png(img: Image.Image) -> str:
    return base64.b64encode(encode(img)).decode()


def multipart_fields(request: httpx.Request) -> dict:
    """Zerlegt einen multipart-Body in {Feldname: Bytes}."""
    boundary = request.headers["content-type"].split("boundary=")[1].encode()
    fields = {}
    for part in request.content.split(b"--" + boundary):
        if b"Content-Disposition" not in part:
            continue
        head, _, body = part.partition(b"\r\n\r\n")
        name = head.split(b'name="')[1].split(b'"')[0].decode()
        fields[name] = body.rstrip(b"\r\n")
    return fields


class Recorder:
    def __init__(self, respond):
        self.respond = respond
        self.requests = []

    def __call__(self, request):
        request.read()
        self.requests.append(request)
        return self.respond(request)


def client_for(respond):
    recorder = Recorder(respond)
    return recorder, httpx.Client(transport=httpx.MockTransport(recorder))


# -- OpenAI ---------------------------------------------------------------------------------
def test_openai_generate_request_and_response():
    rec, http = client_for(lambda r: httpx.Response(200, json={"data": [{"b64_json": b64png(Image.new("RGB", (8, 8), "blue"))}] * 2}))
    images = OpenAIProvider("sk-test", http).generate(GenerateOptions(prompt="Ein Fuchs", aspect_ratio="16:9", quality="high", count=2))
    assert len(images) == 2 and images[0].getpixel((0, 0)) == (0, 0, 255)
    request = rec.requests[0]
    assert request.url == "https://api.openai.com/v1/images/generations"
    assert request.headers["authorization"] == "Bearer sk-test"
    body = json.loads(request.content)
    assert body == {"model": "gpt-image-2.5-flare", "prompt": "Ein Fuchs", "n": 2, "size": "1536x864", "output_format": "png", "quality": "high"}


def test_openai_standard_sizes_for_older_models():
    provider = OpenAIProvider("sk", httpx.Client())
    assert provider.size_for("16:9", "gpt-image-1") == (1536, 1024)
    assert provider.size_for("9:16", "gpt-image-1.5") == (1024, 1536)
    assert provider.size_for("1:1", "gpt-image-1-mini") == (1024, 1024)


def test_openai_auto_quality_is_not_sent():
    rec, http = client_for(lambda r: httpx.Response(200, json={"data": [{"b64_json": b64png(Image.new("RGB", (4, 4)))}]}))
    OpenAIProvider("sk", http).generate(GenerateOptions(prompt="x", quality="auto"))
    assert "quality" not in json.loads(rec.requests[0].content)


def test_openai_edit_sends_image_and_transparent_mask():
    rec, http = client_for(lambda r: httpx.Response(200, json={"data": [{"b64_json": b64png(Image.new("RGB", (1536, 1024), "red"))}]}))
    image = Image.new("RGB", (1536, 1024), "white")
    mask = Image.new("L", (1536, 1024), 0)
    mask.paste(255, (100, 100, 200, 200))
    result = OpenAIProvider("sk", http).edit(image, mask, "Mach einen Hut daraus", "gpt-image-1.5")
    assert result.size == (1536, 1024)
    request = rec.requests[0]
    assert request.url.path == "/v1/images/edits"
    fields = multipart_fields(request)
    assert fields["model"] == b"gpt-image-1.5" and fields["size"] == b"1536x1024"
    sent_mask = Image.open(io.BytesIO(fields["mask"]))
    assert sent_mask.mode == "RGBA" and sent_mask.size == (1536, 1024)
    assert sent_mask.getpixel((150, 150))[3] == 0  # zu bearbeiten = transparent
    assert sent_mask.getpixel((10, 10))[3] == 255


def test_openai_errors_are_translated():
    _, http = client_for(lambda r: httpx.Response(401, json={"error": {"message": "Incorrect API key provided"}}))
    with pytest.raises(ProviderError) as err:
        OpenAIProvider("sk-falsch", http).generate(GenerateOptions(prompt="x"))
    assert err.value.status == 403
    assert "API-Schlüssel ungültig" in str(err.value) and "Incorrect API key" in str(err.value)


def test_network_errors_are_translated():
    def fail(request):
        raise httpx.ConnectError("keine Verbindung")

    _, http = client_for(fail)
    with pytest.raises(ProviderError, match="Keine Verbindung zu OpenAI"):
        OpenAIProvider("sk", http).generate(GenerateOptions(prompt="x"))


# -- Gemini ---------------------------------------------------------------------------------
def gemini_response(img, thought_img=None):
    parts = [{"text": "Hier ist dein Bild"}]
    if thought_img is not None:
        parts.insert(0, {"inlineData": {"mimeType": "image/png", "data": b64png(thought_img)}, "thought": True})
    parts.append({"inlineData": {"mimeType": "image/png", "data": b64png(img)}})
    return {"candidates": [{"content": {"parts": parts, "role": "model"}, "finishReason": "STOP"}]}


def test_gemini_generate():
    final, thought = Image.new("RGB", (16, 9), "green"), Image.new("RGB", (16, 9), "black")
    rec, http = client_for(lambda r: httpx.Response(200, json=gemini_response(final, thought)))
    images = GeminiProvider("g-key", http).generate(GenerateOptions(prompt="Berge", aspect_ratio="16:9", resolution="2K", count=1))
    assert images[0].getpixel((0, 0)) == (0, 128, 0)  # Zwischenbild („thought“) übersprungen
    request = rec.requests[0]
    assert request.url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-nano-banana-2.1:generateContent"
    assert request.headers["x-goog-api-key"] == "g-key"
    body = json.loads(request.content)
    assert body["contents"][0]["parts"] == [{"text": "Berge"}]
    assert body["generationConfig"] == {"responseModalities": ["TEXT", "IMAGE"], "imageConfig": {"aspectRatio": "16:9", "imageSize": "2K"}}


def test_gemini_standard_resolution_is_not_sent():
    rec, http = client_for(lambda r: httpx.Response(200, json=gemini_response(Image.new("RGB", (4, 4)))))
    GeminiProvider("k", http).generate(GenerateOptions(prompt="x", resolution="1K"))
    assert json.loads(rec.requests[0].content)["generationConfig"]["imageConfig"] == {"aspectRatio": "1:1"}


def test_gemini_edit_sends_image_and_mask():
    rec, http = client_for(lambda r: httpx.Response(200, json=gemini_response(Image.new("RGB", (32, 32)))))
    GeminiProvider("k", http).edit(Image.new("RGB", (64, 48)), Image.new("L", (64, 48), 255), "Mehr Sonne", "gemini-3-pro-image")
    body = json.loads(rec.requests[0].content)
    parts = body["contents"][0]["parts"]
    assert parts[0]["text"].startswith("Mehr Sonne") and len(parts) == 3
    assert all(p["inlineData"]["mimeType"] == "image/png" for p in parts[1:])
    assert "gemini-3-pro-image:generateContent" in str(rec.requests[0].url)
    assert body["generationConfig"]["imageConfig"]["aspectRatio"] == "4:3"


def test_gemini_safety_block():
    _, http = client_for(lambda r: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}))
    with pytest.raises(ProviderError, match="Sicherheitsgründen"):
        GeminiProvider("k", http).generate(GenerateOptions(prompt="x"))


def test_gemini_text_only_answer():
    payload = {"candidates": [{"content": {"parts": [{"text": "Das kann ich nicht zeichnen."}]}, "finishReason": "STOP"}]}
    _, http = client_for(lambda r: httpx.Response(200, json=payload))
    with pytest.raises(ProviderError, match="Das kann ich nicht zeichnen"):
        GeminiProvider("k", http).generate(GenerateOptions(prompt="x"))


# -- Stability AI ----------------------------------------------------------------------------
def stability_ok(request):
    return httpx.Response(200, json={"image": b64png(Image.new("RGB", (40, 30), "white")), "finish_reason": "SUCCESS", "seed": 1})


@pytest.mark.parametrize(
    "model,path,model_field",
    [
        ("core", "/v2beta/stable-image/generate/core", None),
        ("ultra", "/v2beta/stable-image/generate/ultra", None),
        ("sd3.5-large", "/v2beta/stable-image/generate/sd3", b"sd3.5-large"),
    ],
)
def test_stability_generate_endpoints(model, path, model_field):
    rec, http = client_for(stability_ok)
    opts = GenerateOptions(prompt="Leuchtturm", model=model, aspect_ratio="3:2", negative_prompt="Text", seed=42, count=2)
    images = StabilityProvider("st-key", http).generate(opts)
    assert len(images) == 2 and len(rec.requests) == 2
    request = rec.requests[0]
    assert request.url.path == path
    assert request.headers["authorization"] == "Bearer st-key" and request.headers["accept"] == "application/json"
    fields = multipart_fields(request)
    assert fields["prompt"] == b"Leuchtturm" and fields["aspect_ratio"] == b"3:2"
    assert fields["negative_prompt"] == b"Text" and fields["seed"] == b"42"
    assert fields.get("model") == model_field


def test_stability_inpaint_and_upscale():
    rec, http = client_for(stability_ok)
    provider = StabilityProvider("k", http)
    provider.edit(Image.new("RGB", (64, 64)), Image.new("L", (64, 64), 255), "Blumen", None)
    provider.upscale(Image.new("RGB", (64, 64)), "fast")
    provider.upscale(Image.new("RGB", (64, 64)), "conservative")
    paths = [r.url.path for r in rec.requests]
    assert paths == [
        "/v2beta/stable-image/edit/inpaint",
        "/v2beta/stable-image/upscale/fast",
        "/v2beta/stable-image/upscale/conservative",
    ]
    inpaint = multipart_fields(rec.requests[0])
    assert inpaint["prompt"] == b"Blumen" and Image.open(io.BytesIO(inpaint["mask"])).mode == "L"
    assert multipart_fields(rec.requests[2])["prompt"]


def test_stability_errors_and_content_filter():
    _, http = client_for(lambda r: httpx.Response(400, json={"name": "bad_request", "errors": ["prompt: zu lang"]}))
    with pytest.raises(ProviderError, match="prompt: zu lang"):
        StabilityProvider("k", http).generate(GenerateOptions(prompt="x"))
    _, http = client_for(lambda r: httpx.Response(200, json={"image": "", "finish_reason": "CONTENT_FILTERED"}))
    with pytest.raises(ProviderError, match="Inhaltsfilter"):
        StabilityProvider("k", http).generate(GenerateOptions(prompt="x"))


# -- Registry ----------------------------------------------------------------------------------
def test_registry_uses_settings_then_environment(tmp_path):
    settings = Settings(config_dir=tmp_path, environ={"GEMINI_API_KEY": "aus-der-umgebung-1234"})
    registry = ProviderRegistry(settings)
    described = {p["id"]: p for p in registry.describe()}
    assert described["gemini"]["configured"] and described["gemini"]["key_source"] == "env"
    assert not described["openai"]["configured"]
    with pytest.raises(ProviderError, match="kein API-Schlüssel"):
        registry.get("openai")
    settings.set_api_key("gemini", "gespeichert-5678")
    assert registry.get("gemini").api_key == "gespeichert-5678"
    assert (tmp_path / "settings.json").exists()


# -- Pipeline ----------------------------------------------------------------------------------
def test_fit_aspect_keeps_box_inside():
    for box, aspect in [((10, 10, 50, 20), 1.0), ((0, 0, 30, 90), 1.5), ((180, 100, 200, 120), 0.5)]:
        out = fit_aspect(box, aspect, 200, 120)
        assert out[0] <= box[0] and out[1] <= box[1] and out[2] >= box[2] and out[3] >= box[3]
        assert 0 <= out[0] and 0 <= out[1] and out[2] <= 200 and out[3] <= 120
    out = fit_aspect((40, 40, 60, 60), 1.5, 200, 120)
    assert abs((out[2] - out[0]) / (out[3] - out[1]) - 1.5) < 0.05


def test_ai_edit_only_changes_the_selection(registry):
    provider = registry.get("fake")
    image = noise_image(300, 200)
    sel = Selection(shapes=[RectShape(x=100, y=60, w=50, h=40)])
    change = ai_edit(provider, image, sel, "Grüner machen", None)
    assert change.kind == "patch" and change.box == (100, 60, 150, 100)
    _, size, mask_size, prompt = FakeProvider.calls[0]
    assert size == mask_size and prompt == "Grüner machen"
    after = image.copy()
    after.paste(change.patch, change.box[:2])
    a, b = np.asarray(image), np.asarray(after)
    outside = np.ones(a.shape[:2], bool)
    outside[60:100, 100:150] = False
    assert np.array_equal(a[outside], b[outside])
    # Ergebnis wird farblich an die Umgebung angepasst, bleibt aber deutlich grün
    inner = b[70:90, 110:140].astype(int)
    assert inner[..., 1].mean() > inner[..., 0].mean() + 20


def test_ai_edit_without_selection_replaces_whole_image(registry):
    change = ai_edit(registry.get("fake"), noise_image(120, 80), None, "Winter", None)
    assert change.kind == "replace" and change.image.size == (120, 80)


def test_ai_enhance_with_upscaler(registry):
    provider = registry.get("fakeup")
    image = noise_image(300, 200)
    sel = Selection(shapes=[RectShape(x=100, y=60, w=80, h=70)])
    change = ai_enhance(provider, image, sel, None, "fast")
    assert change.kind == "patch" and change.box == (100, 60, 180, 130)
    new = ai_enhance(provider, image, sel, None, "fast", as_new=True)
    assert isinstance(new, NewImage) and new.image.size == (320, 280)
    tiny = ai_enhance(provider, image, Selection(shapes=[RectShape(x=0, y=0, w=20, h=20)]), None, "fast", as_new=True)
    assert tiny.image.width >= 80  # sehr kleine Ausschnitte werden vorher auf die Mindestgröße gebracht


def test_color_matching_corrects_small_shifts_only():
    from bildwerk.ai.pipeline import match_colors

    reference = noise_image(80, 80)
    shifted = Image.fromarray(np.clip(np.asarray(reference, dtype=int) + 12, 0, 255).astype(np.uint8))
    keep = np.ones((80, 80), bool)
    keep[20:60, 20:60] = False
    corrected = np.asarray(match_colors(shifted, reference, keep), dtype=int)
    assert abs(corrected[keep].mean() - np.asarray(reference, dtype=int)[keep].mean()) < 2
    unrelated = noise_image(80, 80, seed=9)
    assert match_colors(unrelated, reference, keep) is unrelated


def test_ai_enhance_with_edit_provider_uses_restore_prompt(registry):
    provider = registry.get("fake")
    sel = Selection(shapes=[RectShape(x=10, y=10, w=40, h=30)])
    ai_enhance(provider, noise_image(200, 150), sel, None)
    assert FakeProvider.calls[-1][3] == ENHANCE_PROMPT
    new = ai_enhance(provider, noise_image(200, 150), sel, None, as_new=True)
    assert isinstance(new, NewImage) and new.image.width > 40


def test_ai_endpoints(client, upload):
    generated = client.post("/api/ai/generate", json={"provider": "fake", "prompt": "Ein roter Ballon", "count": 2}).json()
    names = [d["name"] for d in generated["documents"]]
    assert names == ["KI – Ein roter Ballon (1)", "KI – Ein roter Ballon (2)"]

    doc = upload(noise_image(200, 120))["id"]
    selection = {"shapes": [{"type": "ellipse", "x": 40, "y": 30, "w": 80, "h": 50}], "feather": 6}
    result = client.post(f"/api/documents/{doc}/ai/edit", json={"provider": "fake", "prompt": "Grün", "selection": selection}).json()
    preview = result["preview"]
    assert preview["kind"] == "patch" and preview["label"] == "KI: Grün"
    committed = client.post(f"/api/documents/{doc}/commit", json={"preview_id": preview["preview_id"]}).json()
    assert committed["document"]["history"]["labels"][-1] == "KI: Grün"

    enhanced = client.post(f"/api/documents/{doc}/ai/enhance", json={"provider": "fakeup", "upscale_model": "fast", "selection": selection, "new_document": True}).json()
    assert enhanced["new_document"]["name"] == "bild – KI-Detail-Zoom"

    missing = client.post("/api/ai/generate", json={"provider": "gibtsnicht", "prompt": "x"})
    assert missing.status_code == 404
    unsupported = client.post(f"/api/documents/{doc}/ai/enhance", json={"provider": "fake", "new_document": False})
    assert unsupported.status_code == 200  # Edit-Anbieter schärfen per Anweisung
    no_generate = client.post("/api/ai/generate", json={"provider": "fakeup", "prompt": "x"})
    assert no_generate.status_code == 400


def test_ai_enhance_refuses_areas_that_would_get_blurrier(registry):
    from bildwerk.ops import OperationError

    big = Selection(shapes=[RectShape(x=0, y=0, w=600, h=400)])
    with pytest.raises(OperationError, match="zu groß"):
        ai_enhance(registry.get("fake"), noise_image(800, 500), big, None)  # Modell liefert nur 384 × 256

    class SmallUpscaler(FakeUpscaler):
        def upscale_limits(self, model):
            return (32, 100, 10_000)

    with pytest.raises(OperationError, match="zu groß"):
        ai_enhance(SmallUpscaler("k", None), noise_image(800, 500), big, None, "fast")
    assert FakeUpscaler.calls == []  # kein (kostenpflichtiger) Aufruf
    new = ai_enhance(SmallUpscaler("k", None), noise_image(800, 500), big, None, "fast", as_new=True)
    assert isinstance(new, NewImage)
