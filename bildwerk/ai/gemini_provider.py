"""Google Gemini API (Bildmodelle „Nano Banana“)."""

from __future__ import annotations

import base64
import os
from typing import Any

from PIL import Image

from .base import GenerateOptions, ModelOption, Provider, ProviderError, b64_image, closest_ratio, png_bytes

ASPECT_RATIOS = ("1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9")

MASK_HINT = (
    "\n\nThe second image is a mask: change only the area that is white in the mask and keep "
    "everything else exactly as it is. Return the complete image with the same framing and "
    "aspect ratio as the first image."
)


class GeminiProvider(Provider):
    id = "gemini"
    name = "Google Gemini"
    env_vars = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    key_url = "https://aistudio.google.com/apikey"
    capabilities = frozenset({"generate", "edit"})
    models = (
        ModelOption("gemini-nano-banana-2.1", "Nano Banana 2.1", "aktuell, schnell und günstig"),
        ModelOption("gemini-3.1-flash-image", "Nano Banana 2 (Gemini 3.1 Flash Image)"),
        ModelOption("gemini-3-pro-image", "Nano Banana Pro (Gemini 3 Pro Image)", "höchste Qualität"),
        ModelOption("gemini-2.5-flash-image", "Nano Banana (Gemini 2.5 Flash Image)", "ältere Version"),
    )
    default_model = "gemini-nano-banana-2.1"
    aspect_ratios = ASPECT_RATIOS
    resolutions = (("1K", "Standard (1K)"), ("2K", "Hoch (2K)"), ("4K", "Sehr hoch (4K)"))
    max_count = 4
    note = "Höhere Auflösungen (2K/4K) unterstützen nur neuere Modelle."

    @property
    def base_url(self) -> str:
        return os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")

    def edit_size(self, aspect: float, model: str | None) -> tuple[int, int]:
        ratio = closest_ratio(aspect, ASPECT_RATIOS)
        w, _, h = ratio.partition(":")
        scale = 1024 / max(int(w), int(h))
        return round(int(w) * scale), round(int(h) * scale)

    def _call(self, model: str, parts: list[dict[str, Any]], aspect_ratio: str | None, resolution: str | None) -> Image.Image:
        image_config: dict[str, str] = {}
        if aspect_ratio:
            image_config["aspectRatio"] = aspect_ratio
        if resolution and resolution != "1K":
            image_config["imageSize"] = resolution
        generation_config: dict[str, Any] = {"responseModalities": ["TEXT", "IMAGE"]}
        if image_config:
            generation_config["imageConfig"] = image_config
        body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": generation_config}
        response = self.post(
            f"{self.base_url}/models/{model}:generateContent",
            headers={"x-goog-api-key": self.api_key},
            json=body,
        )
        return self._image_from(self.json_of(response))

    def _image_from(self, payload: dict[str, Any]) -> Image.Image:
        texts: list[str] = []
        image_data = None
        reasons = []
        for candidate in payload.get("candidates") or []:
            reasons.append(candidate.get("finishReason"))
            for part in (candidate.get("content") or {}).get("parts") or []:
                if part.get("thought"):
                    continue  # Zwischenbilder des „Nachdenkens“ überspringen
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    image_data = inline["data"]
                elif part.get("text"):
                    texts.append(part["text"])
        if image_data:
            return b64_image(image_data)
        block = (payload.get("promptFeedback") or {}).get("blockReason")
        if block or any(r in ("SAFETY", "PROHIBITED_CONTENT", "IMAGE_SAFETY", "BLOCKLIST") for r in reasons):
            raise ProviderError("Gemini hat die Anfrage aus Sicherheitsgründen abgelehnt.", 400)
        detail = " ".join(texts).strip()[:300]
        raise ProviderError("Gemini hat kein Bild erzeugt." + (f" Antwort: {detail}" if detail else ""), 502)

    def generate(self, opts: GenerateOptions) -> list[Image.Image]:
        model = self.model_or_default(opts.model)
        aspect = opts.aspect_ratio if opts.aspect_ratio in ASPECT_RATIOS else "1:1"
        return [
            self._call(model, [{"text": opts.prompt}], aspect, opts.resolution)
            for _ in range(max(1, min(self.max_count, opts.count)))
        ]

    def edit(self, image: Image.Image, mask: Image.Image, prompt: str, model: str | None) -> Image.Image:
        model = self.model_or_default(model)
        parts: list[dict[str, Any]] = [{"text": prompt + MASK_HINT}, _inline(image.convert("RGB"))]
        parts.append(_inline(mask.convert("L")))
        aspect = closest_ratio(image.width / image.height, ASPECT_RATIOS)
        return self._call(model, parts, aspect, None)


def _inline(img: Image.Image) -> dict[str, Any]:
    return {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(png_bytes(img)).decode("ascii")}}
