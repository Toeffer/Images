"""KI-Anbieter für Bildgenerierung, -bearbeitung und -vergrößerung.

Ein neuer Anbieter ist eine Unterklasse von :class:`Provider` und wird in
``PROVIDER_CLASSES`` eingetragen.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..settings import Settings
from .base import GenerateOptions, ModelOption, Provider, ProviderError
from .comfyui_provider import ComfyUIProvider
from .gemini_provider import GeminiProvider
from .local.models import ModelStore
from .local_provider import LocalProvider
from .openai_provider import OpenAIProvider
from .stability_provider import StabilityProvider

PROVIDER_CLASSES: tuple[type[Provider], ...] = (
    LocalProvider,
    ComfyUIProvider,
    OpenAIProvider,
    GeminiProvider,
    StabilityProvider,
)

MISSING_CAPABILITY = {
    "generate": "kann keine Bilder generieren",
    "edit": "kann keine Bilder bearbeiten",
    "upscale": "kann nicht hochskalieren",
}

__all__ = ["GenerateOptions", "ModelOption", "Provider", "ProviderError", "ProviderRegistry", "PROVIDER_CLASSES"]


class ProviderRegistry:
    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        classes: tuple[type[Provider], ...] = PROVIDER_CLASSES,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.settings = settings
        self.client = client or httpx.Client(follow_redirects=True)
        self.classes = {cls.id: cls for cls in classes}
        # Gemeinsame Ressourcen, die Anbieter über ``self.context`` erreichen
        self.context: dict[str, Any] = {"settings": settings, "http": self.client, **(context or {})}
        self.context.setdefault("local_store", ModelStore(settings.config_dir / "models", self.client))

    def _class(self, provider_id: str) -> type[Provider]:
        try:
            return self.classes[provider_id]
        except KeyError:
            raise ProviderError(f"Unbekannter KI-Anbieter: {provider_id}", 404) from None

    def key_status(self, provider_id: str) -> tuple[str | None, str | None]:
        cls = self._class(provider_id)
        return self.settings.api_key(cls.id, cls.env_vars)

    def describe(self) -> list[dict[str, Any]]:
        result = []
        for cls in self.classes.values():
            info = cls.describe()
            if cls.requires_key:
                key, source = self.settings.api_key(cls.id, cls.env_vars)
                info["configured"] = bool(key)
                info["key_source"] = source
                info["key_hint"] = f"…{key[-4:]}" if key and len(key) > 8 else None
            else:
                info.update({"key_source": None, "key_hint": None})
                info.update(cls.availability(self.context))
            result.append(info)
        return result

    def get(self, provider_id: str, capability: str | None = None) -> Provider:
        cls = self._class(provider_id)
        if capability and capability not in cls.capabilities:
            raise ProviderError(f"{cls.name} {MISSING_CAPABILITY[capability]}.", 400)
        if not cls.requires_key:
            status = cls.availability(self.context)
            if not status.get("configured"):
                raise ProviderError(status.get("status") or f"{cls.name} ist nicht verfügbar.", 400)
            return cls(None, self.client, **self.context)
        key, _ = self.settings.api_key(cls.id, cls.env_vars)
        if not key:
            raise ProviderError(
                f"Für {cls.name} ist noch kein API-Schlüssel hinterlegt. Bitte unter ⚙ Einstellungen eintragen.", 400
            )
        return cls(key, self.client, **self.context)
