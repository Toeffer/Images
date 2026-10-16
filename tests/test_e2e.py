"""Browser-Tests: steuern die echte Oberfläche in Chromium (Playwright).

Ausführen mit ``pytest -m e2e``. Ohne installiertes Playwright/Chromium werden sie übersprungen.
Ein vorhandenes Chromium kann über PLAYWRIGHT_CHROMIUM_EXECUTABLE angegeben werden.
"""

from __future__ import annotations

import io
import os
import socket
import threading
import time

import httpx
import pytest
from PIL import Image, ImageDraw

sync_api = pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.e2e


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    import uvicorn

    from bildwerk.ai import ProviderRegistry
    from bildwerk.server import create_app
    from bildwerk.settings import Settings

    from .conftest import FakeProvider, FakeUpscaler

    settings = Settings(config_dir=tmp_path_factory.mktemp("config"), environ={})
    settings.set_api_key("fake", "test-key-123456")
    registry = ProviderRegistry(settings, classes=(FakeProvider, FakeUpscaler))
    port = _free_port()
    config = uvicorn.Config(create_app(settings=settings, registry=registry), host="127.0.0.1", port=port, log_level="warning")
    uv = uvicorn.Server(config)
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
def browser():
    with sync_api.sync_playwright() as p:
        executable = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
        try:
            instance = p.chromium.launch(executable_path=executable) if executable else p.chromium.launch()
        except Exception as exc:  # pragma: no cover - abhängig von der Umgebung
            pytest.skip(f"Chromium ist nicht verfügbar: {exc}")
        yield instance
        instance.close()


@pytest.fixture
def page(browser, server):
    with httpx.Client(base_url=server) as http:  # sauberer Ausgangszustand
        for doc in http.get("api/documents").json()["documents"]:
            http.delete(f"api/documents/{doc['id']}")
    context = browser.new_context(viewport={"width": 1400, "height": 900}, accept_downloads=True)
    context.add_init_script("delete window.showSaveFilePicker")  # Download statt Speichern-Dialog
    pg = context.new_page()
    errors: list[str] = []
    pg.on("pageerror", lambda exc: errors.append(str(exc)))
    pg.on("console", lambda msg: msg.type == "error" and errors.append(msg.text))
    pg.goto(server)
    pg.wait_for_function("window.bildwerk && window.bildwerk.docs")
    yield pg
    context.close()
    assert not errors, errors


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
