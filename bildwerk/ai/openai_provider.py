"""OpenAI Images API (GPT-Image-Modelle)."""

from __future__ import annotations

import os

from PIL import Image, ImageOps

from .base import GenerateOptions, ModelOption, Provider, ProviderError, b64_image, parse_ratio, png_bytes

# Feste Größen, die alle GPT-Image-Modelle unterstützen
STANDARD_SIZES = {"square": (1024, 1024), "landscape": (1536, 1024), "portrait": (1024, 1536)}
# Ab gpt-image-2 sind beliebige Größen (Vielfache von 16) möglich
FLEXIBLE_SIZES = {
    "1:1": (1024, 1024),
    "3:2": (1536, 1024),
    "2:3": (1024, 1536),
    "4:3": (1408, 1056),
    "3:4": (1056, 1408),
    "16:9": (1536, 864),
    "9:16": (864, 1536),
}


def _standard_size(aspect: float) -> tuple[int, int]:
    if aspect > 1.2:
        return STANDARD_SIZES["landscape"]
    if aspect < 1 / 1.2:
        return STANDARD_SIZES["portrait"]
    return STANDARD_SIZES["square"]


class OpenAIProvider(Provider):
    id = "openai"
    name = "OpenAI"
    env_vars = ("OPENAI_API_KEY",)
    key_url = "https://platform.openai.com/api-keys"
    capabilities = frozenset({"generate", "edit"})
    models = (
        ModelOption("gpt-image-2.5-flare", "GPT Image 2.5 Flare", "schnell, für alltägliche Bilder"),
        ModelOption("gpt-image-2.5-sunburst", "GPT Image 2.5 Sunburst", "präziser, besonders für Bearbeitungen"),
        ModelOption("gpt-image-2", "GPT Image 2"),
        ModelOption("gpt-image-1.5", "GPT Image 1.5"),
        ModelOption("gpt-image-1-mini", "GPT Image 1 mini", "günstig"),
        ModelOption("gpt-image-1", "GPT Image 1"),
    )
    default_model = "gpt-image-2.5-flare"
    aspect_ratios = tuple(FLEXIBLE_SIZES)
    qualities = (("auto", "Automatisch"), ("low", "Niedrig (günstig)"), ("medium", "Mittel"), ("high", "Hoch"))
    max_count = 4
    note = "Manche Modelle setzen eine verifizierte Organisation im OpenAI-Konto voraus."

    @property
    def base_url(self) -> str:
        return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _flexible(model: str) -> bool:
        return model.startswith("gpt-image-2")

    def size_for(self, aspect_ratio: str, model: str) -> tuple[int, int]:
        if self._flexible(model) and aspect_ratio in FLEXIBLE_SIZES:
            return FLEXIBLE_SIZES[aspect_ratio]
        return _standard_size(parse_ratio(aspect_ratio))

    def edit_size(self, aspect: float, model: str | None) -> tuple[int, int]:
        return _standard_size(aspect)

    def _images_from(self, payload: dict) -> list[Image.Image]:
        items = payload.get("data") or []
        images = [b64_image(item["b64_json"]) for item in items if item.get("b64_json")]
        if not images:
            raise ProviderError("OpenAI hat kein Bild zurückgegeben.")
        return images

    def generate(self, opts: GenerateOptions) -> list[Image.Image]:
        model = self.model_or_default(opts.model)
        width, height = self.size_for(opts.aspect_ratio, model)
        body: dict = {
            "model": model,
            "prompt": opts.prompt,
            "n": max(1, min(self.max_count, opts.count)),
            "size": f"{width}x{height}",
            "output_format": "png",
        }
        if opts.quality and opts.quality != "auto":
            body["quality"] = opts.quality
        response = self.post(f"{self.base_url}/images/generations", headers=self.headers, json=body)
        return self._images_from(self.json_of(response))

    def edit(self, image: Image.Image, mask: Image.Image, prompt: str, model: str | None) -> Image.Image:
        model = self.model_or_default(model)
        # OpenAI erwartet eine PNG-Maske, deren transparente Pixel den zu bearbeitenden Bereich markieren.
        alpha = ImageOps.invert(mask.convert("L"))
        mask_rgba = Image.new("RGBA", image.size, (0, 0, 0, 255))
        mask_rgba.putalpha(alpha)
        files = {
            "image": ("image.png", png_bytes(image.convert("RGB")), "image/png"),
            "mask": ("mask.png", png_bytes(mask_rgba), "image/png"),
        }
        data = {
            "model": model,
            "prompt": prompt,
            "n": "1",
            "size": f"{image.width}x{image.height}",
            "output_format": "png",
        }
        response = self.post(f"{self.base_url}/images/edits", headers=self.headers, files=files, data=data)
        return self._images_from(self.json_of(response))[0]
