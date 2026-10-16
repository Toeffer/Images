import io

import numpy as np
from PIL import Image

from .conftest import encode, noise_image

RECT = {"shapes": [{"type": "rect", "x": 10, "y": 10, "w": 30, "h": 20}], "feather": 0, "invert": False}


def test_info_lists_operations_and_providers(client):
    data = client.get("/api/info").json()
    assert {c["id"] for c in data["categories"]} == {"adjust", "filter", "paint", "transform"}
    assert [p["id"] for p in data["providers"]] == ["fake", "fakeup"]
    assert {f["id"] for f in data["export_formats"]} >= {"png", "jpeg", "webp"}


def test_index_and_static_files(client):
    assert "Bildwerk" in client.get("/").text
    assert client.get("/static/js/main.js").status_code == 200


def test_upload_applies_exif_orientation(client):
    img = noise_image(60, 40)
    exif = Image.Exif()
    exif[0x0112] = 6  # um 90° gedreht aufgenommen
    response = client.post("/api/documents", files={"file": ("handy.jpg", encode(img, "JPEG", exif=exif), "image/jpeg")})
    info = response.json()
    assert (info["width"], info["height"]) == (40, 60)
    assert info["source_format"] == "JPEG" and info["dirty"] is False


def test_upload_rejects_non_images(client):
    response = client.post("/api/documents", files={"file": ("notiz.txt", b"kein Bild", "text/plain")})
    assert response.status_code == 400
    assert "nicht unterstützt" in response.json()["detail"]


def test_upload_normalizes_16bit_grayscale(client, upload):
    arr = (np.linspace(0, 65535, 64 * 32).reshape(32, 64)).astype(np.uint16)
    scan = Image.fromarray(arr)
    assert scan.mode.startswith("I;16")
    info = upload(scan, "scan.png")
    raw = client.get(f"/api/documents/{info['id']}/pixels").content
    pixels = np.frombuffer(raw, np.uint8).reshape(32, 64, 4)
    assert pixels[0, 0, 0] == 0 and pixels[-1, -1, 0] == 255


def test_pixels_full_and_region(client, upload):
    img = noise_image(50, 40)
    info = upload(img)
    full = client.get(f"/api/documents/{info['id']}/pixels")
    assert full.headers["x-width"] == "50" and len(full.content) == 50 * 40 * 4
    region = client.get(f"/api/documents/{info['id']}/pixels?x=10&y=5&w=8&h=6")
    rgba = np.frombuffer(region.content, np.uint8).reshape(6, 8, 4)
    assert np.array_equal(rgba[..., :3], np.asarray(img)[5:11, 10:18])
    assert (rgba[..., 3] == 255).all()


def test_preview_commit_and_undo(client, upload):
    doc = upload()["id"]
    preview = client.post(f"/api/documents/{doc}/preview", json={"op": "invert", "selection": RECT}).json()
    assert preview["kind"] == "patch" and preview["box"] == [10, 10, 40, 30]
    patch = client.get(f"/api/documents/{doc}/preview/{preview['preview_id']}/pixels")
    assert len(patch.content) == 30 * 20 * 4
    committed = client.post(f"/api/documents/{doc}/commit", json={"preview_id": preview["preview_id"]}).json()
    assert committed["change"] == {"kind": "patch", "box": [10, 10, 40, 30]}
    assert committed["document"]["history"]["labels"] == ["Geöffnet", "Invertieren"]
    again = client.post(f"/api/documents/{doc}/commit", json={"preview_id": preview["preview_id"]})
    assert again.status_code == 409
    undone = client.post(f"/api/documents/{doc}/undo").json()
    assert undone["document"]["history"]["position"] == 0


def test_apply_reuses_matching_preview(client, upload):
    doc = upload()["id"]
    request = {"op": "sharpen", "params": {"amount": 150}, "selection": RECT}
    client.post(f"/api/documents/{doc}/preview", json=request)
    preview_pixels = client.get(f"/api/documents/{doc}/pixels").content
    applied = client.post(f"/api/documents/{doc}/apply", json=request).json()
    assert applied["change"]["kind"] == "patch"
    assert client.get(f"/api/documents/{doc}/pixels").content != preview_pixels


def test_apply_transform_and_new_document(client, upload):
    doc = upload(noise_image(60, 40))["id"]
    rotated = client.post(f"/api/documents/{doc}/apply", json={"op": "rotate90", "params": {"direction": "ccw"}}).json()
    assert rotated["change"] == {"kind": "replace"}
    assert (rotated["document"]["width"], rotated["document"]["height"]) == (40, 60)
    zoom = client.post(f"/api/documents/{doc}/apply", json={"op": "detail_zoom", "params": {"factor": "2"}, "selection": RECT}).json()
    new = zoom["new_document"]
    assert (new["width"], new["height"]) == (60, 40)
    assert new["name"] == "bild – Detail-Zoom" and new["dirty"] is True
    assert len(client.get("/api/documents").json()["documents"]) == 2


def test_operation_errors_are_readable(client, upload):
    doc = upload()["id"]
    assert client.post(f"/api/documents/{doc}/apply", json={"op": "crop"}).json()["detail"] == "Bitte zuerst einen Bereich auswählen."
    assert client.post(f"/api/documents/{doc}/apply", json={"op": "gibtsnicht"}).status_code == 400
    assert client.post(f"/api/documents/{doc}/apply", json={"op": "rotate90", "params": {"direction": "quer"}}).status_code == 400
    assert client.get("/api/documents/unbekannt").status_code == 404


def test_history_endpoint(client, upload):
    doc = upload()["id"]
    for op in ("invert", "grayscale", "sepia"):
        client.post(f"/api/documents/{doc}/apply", json={"op": op})
    result = client.post(f"/api/documents/{doc}/history/1").json()
    assert result["document"]["history"]["position"] == 1
    assert result["change"]["kind"] == "replace"


def test_export_formats_and_saved_flag(client, upload):
    doc = upload(noise_image(30, 20, "RGBA"))["id"]
    client.post(f"/api/documents/{doc}/apply", json={"op": "invert"})
    assert client.get(f"/api/documents/{doc}").json()["dirty"] is True
    for fmt, mime in [("png", "image/png"), ("jpeg", "image/jpeg"), ("webp", "image/webp"), ("tiff", "image/tiff")]:
        response = client.get(f"/api/documents/{doc}/export?format={fmt}&quality=80")
        assert response.status_code == 200 and response.headers["content-type"] == mime
        decoded = Image.open(io.BytesIO(response.content))
        assert decoded.size == (30, 20)
        assert "attachment" in response.headers["content-disposition"]
    assert client.get(f"/api/documents/{doc}").json()["dirty"] is False
    assert client.get(f"/api/documents/{doc}/export?format=bmp").status_code == 400


def test_jpeg_export_flattens_transparency_on_white(client):
    response = client.post("/api/documents/new", json={"width": 10, "height": 10, "background": "transparent"})
    doc = response.json()["id"]
    data = client.get(f"/api/documents/{doc}/export?format=jpeg").content
    assert Image.open(io.BytesIO(data)).convert("RGB").getpixel((5, 5)) == (255, 255, 255)


def test_new_rename_close(client):
    info = client.post("/api/documents/new", json={"width": 30, "height": 20, "background": "#ff0000", "name": "Leer"}).json()
    assert info["has_alpha"] is False and info["history"]["labels"] == ["Neues Bild"]
    assert client.patch(f"/api/documents/{info['id']}", json={"name": "Plakat"}).json()["name"] == "Plakat"
    assert client.delete(f"/api/documents/{info['id']}").json() == {"ok": True}
    assert client.get(f"/api/documents/{info['id']}").status_code == 404
    bad = client.post("/api/documents/new", json={"width": 30, "height": 20, "background": "rot"})
    assert bad.status_code == 422


def test_settings_store_keys_without_leaking_them(client, settings):
    response = client.put("/api/settings/keys/fake", json={"api_key": "geheim-schluessel-9876"})
    providers = {p["id"]: p for p in response.json()["providers"]}
    assert providers["fake"]["key_hint"] == "…9876" and providers["fake"]["key_source"] == "settings"
    assert "geheim-schluessel-9876" not in client.get("/api/settings").text
    assert "geheim-schluessel-9876" not in client.get("/api/info").text
    client.put("/api/settings/keys/fake", json={"api_key": ""})
    assert settings.api_key("fake", ()) == (None, None)
    assert client.put("/api/settings/keys/gibtsnicht", json={"api_key": "x"}).status_code == 404


def test_foreign_origin_and_host_are_rejected(client, settings, registry):
    from fastapi.testclient import TestClient

    from bildwerk.server import create_app

    blocked = client.post("/api/documents/new", json={"width": 5, "height": 5}, headers={"Origin": "https://boese.example"})
    assert blocked.status_code == 403
    same = client.post("/api/documents/new", json={"width": 5, "height": 5}, headers={"Origin": "http://testserver"})
    assert same.status_code == 200
    local_only = TestClient(create_app(settings=settings, registry=registry))
    assert local_only.get("/api/info").status_code == 400  # Host "testserver" ist nicht erlaubt


def test_heic_without_plugin_gives_helpful_hint(client, monkeypatch):
    import bildwerk.imageio as imageio

    monkeypatch.setattr(imageio, "HEIC_SUPPORT", False)
    fake_heic = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64
    response = client.post("/api/documents", files={"file": ("IMG_0001.HEIC", fake_heic, "image/heic")})
    assert response.status_code == 400 and "pillow-heif" in response.json()["detail"]
