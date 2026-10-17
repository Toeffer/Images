"""Lokale KI direkt in Bildwerk: Real-ESRGAN über ONNX Runtime (Grafikkarte oder CPU)."""

from __future__ import annotations

from typing import Any

from PIL import Image

from .base import ModelOption, Provider, ProviderError
from .local.models import CATALOG, ModelError, ModelStore
from .local.runtime import onnxruntime_available


def _mb(size: float) -> str:
    return f"{size:.1f} MB".replace(".", ",")


class LocalProvider(Provider):
    id = "local"
    name = "Lokal auf diesem PC"
    env_vars = ()
    key_url = "https://github.com/xinntao/Real-ESRGAN"
    capabilities = frozenset({"upscale"})
    models = ()
    default_model = ""
    upscale_models = tuple(
        ModelOption(m.id, m.label, f"{m.description} Einmaliger Download: {_mb(m.size_mb)}.") for m in CATALOG.values()
    )
    requires_key = False
    local = True
    note = "Kostenlos und offline – das Modell wird beim ersten Mal heruntergeladen."

    @property
    def store(self) -> ModelStore:
        return self.context["local_store"]

    @classmethod
    def availability(cls, context: dict[str, Any]) -> dict[str, Any]:
        if not onnxruntime_available():
            return {
                "configured": False,
                "status": "Paket „onnxruntime“ fehlt – start.bat erneut ausführen. Klappt es nicht, Python 3.12 bis 3.14 verwenden.",
            }
        store: ModelStore = context["local_store"]
        upscale_models = []
        for m in CATALOG.values():
            installed = store.installed(m)
            extra = "Bereits geladen." if installed else f"Einmaliger Download: {_mb(m.size_mb)}."
            upscale_models.append({"id": m.id, "label": m.label, "description": f"{m.description} {extra}", "installed": installed})
        count = sum(m["installed"] for m in upscale_models)
        models = f"{count} von {len(CATALOG)} Modellen geladen" if count else "Modelle werden bei Bedarf geladen"
        return {"configured": True, "status": f"Rechnet auf: {store.device()} · {models}", "upscale_models": upscale_models}

    def upscale_limits(self, model: str | None) -> tuple[int, int, int]:
        # Kein Limit des Anbieters – nur die Rechenzeit wächst mit der Größe
        return (16, 4096, 4_000_000)

    def upscale(self, image: Image.Image, model: str | None) -> Image.Image:
        spec = CATALOG.get(model or "general")
        if spec is None:
            raise ProviderError(f"Unbekanntes lokales Modell: {model}", 400)
        try:
            return self.store.upscale(spec, image)
        except ModelError as exc:
            raise ProviderError(str(exc), 500) from exc
