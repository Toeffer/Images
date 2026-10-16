"""Stability AI REST API (Stable Image, Stable Diffusion 3.5)."""

from __future__ import annotations

import math
import os
from typing import Any

from PIL import Image

from .base import GenerateOptions, ModelOption, Provider, ProviderError, b64_image, png_bytes

ASPECT_RATIOS = ("1:1", "3:2", "2:3", "5:4", "4:5", "16:9", "9:16", "21:9", "9:21")
SD3_MODELS = ("sd3.5-large", "sd3.5-large-turbo", "sd3.5-medium")


class StabilityProvider(Provider):
    id = "stability"
    name = "Stability AI"
    env_vars = ("STABILITY_API_KEY",)
    key_url = "https://platform.stability.ai/account/keys"
    capabilities = frozenset({"generate", "edit", "upscale"})
    models = (
        ModelOption("core", "Stable Image Core", "schnell und günstig"),
        ModelOption("ultra", "Stable Image Ultra", "höchste Qualität"),
        ModelOption("sd3.5-large", "Stable Diffusion 3.5 Large"),
        ModelOption("sd3.5-large-turbo", "Stable Diffusion 3.5 Large Turbo", "schnell"),
        ModelOption("sd3.5-medium", "Stable Diffusion 3.5 Medium"),
    )
    default_model = "core"
    aspect_ratios = ASPECT_RATIOS
    upscale_models = (
        ModelOption("fast", "Schnell (4×)", "günstig, sehr schnell"),
        ModelOption("conservative", "Detailtreu (bis 4K)", "langsamer, mehr Details"),
    )
    supports_negative_prompt = True
    supports_seed = True
    max_count = 4

    @property
    def base_url(self) -> str:
        return os.environ.get("STABILITY_BASE_URL", "https://api.stability.ai").rstrip("/")

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}

    def extract_error(self, payload: Any) -> str | None:
        if isinstance(payload, dict) and isinstance(payload.get("errors"), list):
            return "; ".join(str(e) for e in payload["errors"])
        return super().extract_error(payload)

    def edit_size(self, aspect: float, model: str | None) -> tuple[int, int]:
        # rund 1 Megapixel, Seiten als Vielfache von 64
        height = math.sqrt(1024 * 1024 / aspect)
        width = height * aspect
        return max(64, round(width / 64) * 64), max(64, round(height / 64) * 64)

    def upscale_limits(self, model: str | None) -> tuple[int, int, int]:
        if model == "conservative":
            return (64, 4096, 9_000_000)
        return (32, 1536, 1_048_576)

    def _request(self, path: str, fields: dict[str, Any], images: dict[str, Image.Image] | None = None) -> Image.Image:
        # Stability erwartet immer multipart/form-data, auch ohne Bilddatei.
        files: dict[str, Any] = {k: (None, str(v)) for k, v in fields.items() if v is not None and v != ""}
        for name, img in (images or {}).items():
            files[name] = (f"{name}.png", png_bytes(img), "image/png")
        response = self.post(f"{self.base_url}{path}", headers=self.headers, files=files)
        payload = self.json_of(response)
        if payload.get("finish_reason") == "CONTENT_FILTERED":
            raise ProviderError("Stability AI hat das Ergebnis durch den Inhaltsfilter blockiert.", 400)
        if not payload.get("image"):
            raise ProviderError("Stability AI hat kein Bild zurückgegeben.")
        return b64_image(payload["image"])

    def generate(self, opts: GenerateOptions) -> list[Image.Image]:
        model = self.model_or_default(opts.model)
        if model in SD3_MODELS:
            path, extra = "/v2beta/stable-image/generate/sd3", {"model": model}
        elif model in ("core", "ultra"):
            path, extra = f"/v2beta/stable-image/generate/{model}", {}
        else:
            raise ProviderError(f"Unbekanntes Stability-Modell: {model}", 400)
        fields = {
            "prompt": opts.prompt,
            "aspect_ratio": opts.aspect_ratio if opts.aspect_ratio in ASPECT_RATIOS else "1:1",
            "output_format": "png",
            "negative_prompt": opts.negative_prompt,
            "seed": opts.seed,
            **extra,
        }
        return [self._request(path, fields) for _ in range(max(1, min(self.max_count, opts.count)))]

    def edit(self, image: Image.Image, mask: Image.Image, prompt: str, model: str | None) -> Image.Image:
        fields = {"prompt": prompt, "output_format": "png", "grow_mask": 5}
        return self._request(
            "/v2beta/stable-image/edit/inpaint", fields, {"image": image.convert("RGB"), "mask": mask.convert("L")}
        )

    def upscale(self, image: Image.Image, model: str | None) -> Image.Image:
        if model == "conservative":
            fields = {"prompt": "high quality, sharp, detailed photograph", "output_format": "png"}
            return self._request("/v2beta/stable-image/upscale/conservative", fields, {"image": image.convert("RGB")})
        return self._request("/v2beta/stable-image/upscale/fast", {"output_format": "png"}, {"image": image.convert("RGB")})
