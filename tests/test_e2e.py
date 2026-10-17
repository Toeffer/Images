"""Browser-Tests: steuern die echte Oberfläche in Chromium (Playwright).

Ausführen mit ``pytest -m e2e``. Ohne installiertes Playwright/Chromium werden sie übersprungen.
Ein vorhandenes Chromium kann über PLAYWRIGHT_CHROMIUM_EXECUTABLE angegeben werden.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
from PIL import Image, ImageDraw

sync_api = pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.e2e


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def serve(app):
    import uvicorn

    port = _free_port()
    uv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not uv.started:
        if time.time() > deadline:
            pytest.fail("Server ist nicht gestartet")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/"
    uv.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    from bildwerk.ai import ProviderRegistry
    from bildwerk.server import create_app
    from bildwerk.settings import Settings

    from .conftest import FakeProvider, FakeUpscaler

    settings = Settings(config_dir=tmp_path_factory.mktemp("config"), environ={})
    settings.set_api_key("fake", "test-key-123456")
    registry = ProviderRegistry(settings, classes=(FakeProvider, FakeUpscaler))
    with serve(create_app(settings=settings, registry=registry)) as url:
        yield url


@pytest.fixture(scope="module")
def local_server(tmp_path_factory):
    """Bildwerk mit lokalem Anbieter (Mini-Gewichte) und simuliertem ComfyUI."""
    from bildwerk.ai import ProviderRegistry
    from bildwerk.ai.comfyui_provider import ComfyUIProvider
    from bildwerk.ai.local_provider import LocalProvider
    from bildwerk.server import create_app
    from bildwerk.settings import Settings

    from .test_comfyui import FakeComfy

    pytest.importorskip("onnxruntime")
    fixtures = Path(__file__).parent / "fixtures"
    weights = {"realesr-general-x4v3.pth": "tiny_srvgg.pth", "realesr-general-wdn-x4v3.pth": "tiny_srvgg_b.pth"}
    comfy = FakeComfy(pending_polls=1)

    def transport(request):
        name = request.url.path.rsplit("/", 1)[-1]
        if name in weights:
            return httpx.Response(200, content=(fixtures / weights[name]).read_bytes())
        if request.url.host == "comfy.test":
            return comfy(request)
        raise httpx.ConnectError("im Test nicht erreichbar")

    settings = Settings(config_dir=tmp_path_factory.mktemp("config"), environ={})
    settings.set("comfyui_url", "http://comfy.test:8188")
    client = httpx.Client(transport=httpx.MockTransport(transport))
    registry = ProviderRegistry(settings, client=client, classes=(LocalProvider, ComfyUIProvider))
    with serve(create_app(settings=settings, registry=registry)) as url:
        yield url, comfy


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        executable = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
        try:
            instance = p.chromium.launch(executable_path=executable) if executable else p.chromium.launch()
        except Exception as exc:  # pragma: no cover - abhängig von der Umgebung
            pytest.skip(f"Chromium ist nicht verfügbar: {exc}")
        yield instance
        instance.close()


@contextlib.contextmanager
def open_app(browser, url):
    with httpx.Client(base_url=url) as http:  # sauberer Ausgangszustand
        for doc in http.get("api/documents").json()["documents"]:
            http.delete(f"api/documents/{doc['id']}")
    context = browser.new_context(viewport={"width": 1400, "height": 900}, accept_downloads=True)
    context.add_init_script("delete window.showSaveFilePicker")  # Download statt Speichern-Dialog
    pg = context.new_page()
    errors: list[str] = []
    pg.on("pageerror", lambda exc: errors.append(str(exc)))
    pg.on("console", lambda msg: msg.type == "error" and errors.append(msg.text))
    pg.goto(url)
    pg.wait_for_function("window.bildwerk && window.bildwerk.docs")
    yield pg
    context.close()
    assert not errors, errors


@pytest.fixture
def page(browser, server):
    with open_app(browser, server) as pg:
        yield pg


@pytest.fixture
def local_page(browser, local_server):
    from bildwerk.ai import comfyui_provider

    comfyui_provider._probe_cache.clear()
    with open_app(browser, local_server[0]) as pg:
        yield pg


@pytest.fixture
def sample(tmp_path):
    img = Image.new("RGB", (800, 500), (70, 120, 200))
    draw = ImageDraw.Draw(img)
    draw.ellipse([500, 60, 640, 200], fill=(250, 210, 120))
    draw.rectangle([0, 330, 800, 500], fill=(50, 110, 60))
    draw.text((60, 60), "Bildwerk", fill="white")
    path = tmp_path / "landschaft.png"
    img.save(path)
    return path


def open_image(page, path):
    page.set_input_files("#file-input", str(path))
    page.wait_for_selector(".tab.active")
    page.wait_for_function("bildwerk.docs.active && bildwerk.docs.active.canvas.width > 0")


def drag(page, start, end):
    box = page.locator("#canvas-overlay").bounding_box()
    page.mouse.move(box["x"] + start[0], box["y"] + start[1])
    page.mouse.down()
    page.mouse.move(box["x"] + end[0], box["y"] + end[1], steps=8)
    page.mouse.up()


def pixel(page, x, y):
    return page.evaluate(f"Array.from(bildwerk.docs.active.canvas.getContext('2d').getImageData({x}, {y}, 1, 1).data)")


def test_open_select_adjust_undo(page, sample):
    open_image(page, sample)
    assert page.locator(".tab.active .tab-name").inner_text() == "landschaft.png"
    drag(page, (200, 150), (420, 330))
    assert page.locator("#status-selection").inner_text().startswith("Auswahl:")

    page.click("[data-op=brightness_contrast]")
    page.locator(".field input[type=range]").first.fill("60")
    page.wait_for_function("bildwerk.docs.active.preview?.source === 'op'")
    selection = page.evaluate("bildwerk.docs.active.selection.bounds(bildwerk.docs.active.width, bildwerk.docs.active.height)")
    cx, cy = (selection[0] + selection[2]) // 2, (selection[1] + selection[3]) // 2
    before, outside_before = pixel(page, cx, cy), pixel(page, 5, 5)
    page.click(".editor-actions .btn.primary")
    page.wait_for_function("bildwerk.docs.active.info.history.labels.length === 2")
    assert pixel(page, cx, cy)[0] > before[0] + 30  # in der Auswahl heller geworden
    assert pixel(page, 5, 5) == outside_before  # außerhalb unverändert

    page.keyboard.press("Control+z")
    page.wait_for_function("bildwerk.docs.active.info.history.position === 0")
    assert pixel(page, cx, cy) == before
    page.click(".sidebar-tabs [data-panel=history]")
    assert page.locator(".history-list li").count() == 2


def test_ai_generate_and_edit(page, sample):
    page.click(".sidebar-tabs [data-panel=ai]")
    page.fill("#panel-ai textarea >> nth=0", "Ein roter Ballon")
    page.click("#panel-ai .btn.ai >> nth=0")
    page.wait_for_selector(".tab.active:has-text('KI – Ein roter Ballon')")

    open_image(page, sample)
    drag(page, (150, 120), (300, 260))
    page.fill("#panel-ai textarea >> nth=1", "Mach es grün")
    page.click("#panel-ai .btn.ai >> nth=1")
    page.wait_for_selector("#result-bar:not([hidden])")
    assert "KI: Mach es grün" in page.locator("#result-label").inner_text()
    page.click("#result-accept")
    page.wait_for_selector("#result-bar", state="hidden")
    assert page.evaluate("bildwerk.docs.active.info.history.labels.at(-1)") == "KI: Mach es grün"


def test_new_image_crop_and_export(page):
    page.click("#welcome [data-command=new]")
    page.select_option("#dlg-new select[name=preset]", "custom")
    page.fill("#dlg-new input[name=width]", "400")
    page.fill("#dlg-new input[name=height]", "300")
    page.check("#dlg-new input[name=bg][value='#000000']")
    page.click("#dlg-new button[value=ok]")
    page.wait_for_function("bildwerk.docs.active && bildwerk.docs.active.width === 400")

    drag(page, (300, 200), (500, 350))
    page.keyboard.press("c")  # Kürzel für Zuschneiden
    page.click(".editor-actions .btn.primary")
    page.wait_for_function("bildwerk.docs.active.width < 400")
    width, height = page.evaluate("[bildwerk.docs.active.width, bildwerk.docs.active.height]")
    assert 0 < width < 400 and 0 < height < 300

    page.keyboard.press("Control+s")
    page.select_option("#dlg-export select[name=format]", "png")
    with page.expect_download() as download:
        page.click("#dlg-export button[value=ok]")
    data = open(download.value.path(), "rb").read()
    assert Image.open(io.BytesIO(data)).size == (width, height)
    page.wait_for_function("!document.querySelector('.tab.active .dirty')")


def test_settings_dialog_configures_provider(page):
    page.click("[data-command=settings]")
    page.fill("#settings-providers .provider-card >> nth=1 >> input", "neuer-schluessel-4321")
    page.click("#settings-providers .provider-card >> nth=1 >> button:has-text('Speichern')")
    page.wait_for_selector("#settings-providers .provider-card >> nth=1 >> text=…4321")
    page.click("#dlg-settings button[value=close]")
    page.click(".sidebar-tabs [data-panel=ai]")
    page.select_option("#panel-ai select >> nth=0", "fakeup")
    assert "Bereit" in page.locator("#panel-ai .ai-status").inner_text()


def test_local_enhance_downloads_model(local_page, sample):
    page = local_page
    page.click(".sidebar-tabs [data-panel=ai]")
    page.select_option("#panel-ai select >> nth=0", "local")
    assert "Einmaliger Download" in page.locator("#panel-ai .card >> nth=1").inner_text()
    open_image(page, sample)
    drag(page, (150, 120), (300, 260))
    page.click("#panel-ai .btn.ai")
    page.wait_for_selector("#result-bar:not([hidden])", timeout=30000)
    page.click("#result-accept")
    page.wait_for_selector("#result-bar", state="hidden")
    assert page.evaluate("bildwerk.docs.active.info.history.labels.at(-1)") == "KI-Schärfen"
    page.wait_for_selector("#panel-ai .card >> text=Bereits geladen")  # Anzeige nach dem Download aktualisiert


def test_comfyui_generate_and_workflow_import(local_page, local_server, tmp_path):
    from .test_comfyui import ZIMAGE

    page, comfy = local_page, local_server[1]
    page.click(".sidebar-tabs [data-panel=ai]")
    page.select_option("#panel-ai select >> nth=0", "comfyui")
    assert "RTX 5060" in page.locator("#panel-ai .ai-status").inner_text()
    page.fill("#panel-ai .card >> nth=1 >> input[type=number] >> nth=1", "12")  # Schritte
    page.fill("#panel-ai textarea >> nth=0", "Ein Leuchtturm")
    page.click("#panel-ai .btn.ai >> nth=0")
    page.wait_for_selector(".tab.active:has-text('KI – Ein Leuchtturm')")
    graph = comfy.graphs[-1]
    assert graph["2"]["inputs"]["text"] == "Ein Leuchtturm" and graph["5"]["inputs"]["steps"] == 12

    workflow = tmp_path / "Z-Image Turbo.json"
    workflow.write_text(json.dumps(ZIMAGE))
    page.click("[data-command=settings]")
    page.wait_for_selector("#settings-local .provider-card >> text=Verbunden")
    page.set_input_files("#settings-local input[type=file]", str(workflow))
    page.wait_for_selector("#settings-local .model-row:has-text('Z-Image Turbo')")
    page.click("#dlg-settings button[value=close]")
    assert page.locator("#panel-ai .card >> nth=1 >> select[aria-label=Modell]").input_value().startswith("wf:z-image-turbo")
    page.fill("#panel-ai textarea >> nth=0", "Ein Fuchs im Schnee")
    page.click("#panel-ai .btn.ai >> nth=0")
    page.wait_for_selector(".tab.active:has-text('KI – Ein Fuchs im Schnee')")
    assert comfy.graphs[-1]["45"]["inputs"]["text"] == "Ein Fuchs im Schnee"
