from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from bildwerk.ai import ProviderRegistry
from bildwerk.ai.base import GenerateOptions, ModelOption, Provider
from bildwerk.server import create_app
from bildwerk.settings import Settings


def noise_image(width: int = 120, height: int = 80, mode: str = "RGB", seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    bands = len(mode)
    arr = (rng.random((height, width, bands)) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode)


def encode(img: Image.Image, fmt: str = "PNG", **kwargs) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, **kwargs)
    return buf.getvalue()


class FakeProvider(Provider):
    """KI-Anbieter für Tests: arbeitet lokal und protokolliert die Aufrufe."""

    id = "fake"
    name = "Testanbieter"
    env_vars = ("FAKE_API_KEY",)
    key_url = "https://example.invalid/keys"
    capabilities = frozenset({"generate", "edit"})
    models = (ModelOption("fake-1", "Fake 1"),)
    default_model = "fake-1"
    calls: list = []

    def edit_size(self, aspect, model):
        return (256, 256) if 0.8 < aspect < 1.25 else (384, 256) if aspect > 1 else (256, 384)

    def generate(self, opts: GenerateOptions):
        type(self).calls.append(("generate", opts))
        return [Image.new("RGB", (64, 48), (200, 30, 30)) for _ in range(opts.count)]

    def edit(self, image, mask, prompt, model):
        type(self).calls.append(("edit", image.size, mask.size, prompt))
        return Image.new("RGB", image.size, (0, 255, 0))


class FakeUpscaler(FakeProvider):
    id = "fakeup"
    name = "Test-Hochskalierer"
    capabilities = frozenset({"upscale"})

    def upscale(self, image, model):
        type(self).calls.append(("upscale", image.size, model))
        return image.resize((image.width * 4, image.height * 4), Image.Resampling.NEAREST)


@pytest.fixture
def settings(tmp_path):
    return Settings(config_dir=tmp_path / "config", environ={})


@pytest.fixture
def registry(settings):
    FakeProvider.calls = []
    FakeUpscaler.calls = []
    settings.set_api_key("fake", "test-key-123456")
    settings.set_api_key("fakeup", "test-key-654321")
    return ProviderRegistry(settings, classes=(FakeProvider, FakeUpscaler))


@pytest.fixture
def client(settings, registry):
    app = create_app(settings=settings, registry=registry, allowed_hosts=["testserver"])
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def upload(client):
    def _upload(img: Image.Image | None = None, name: str = "bild.png", fmt: str = "PNG", **kwargs) -> dict:
        img = img if img is not None else noise_image()
        response = client.post("/api/documents", files={"file": (name, encode(img, fmt, **kwargs), "image/png")})
        assert response.status_code == 200, response.text
        return response.json()

    return _upload
