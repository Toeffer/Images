"""KI-Anbieter für Bildgenerierung, -bearbeitung und -vergrößerung.

Ein neuer Anbieter ist eine Unterklasse von :class:`Provider` und wird in
``PROVIDER_CLASSES`` eingetragen.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..settings import Settings
from .base import GenerateOptions, ModelOption, Provider, ProviderError
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider
from .stability_provider import StabilityProvider

PROVIDER_CLASSES: tuple[type[Provider], ...] = (OpenAIProvider, GeminiProvider, StabilityProvider)

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
    ) -> None:
        self.settings = settings
        self.client = client or httpx.Client(follow_redirects=True)
        self.classes = {cls.id: cls for cls in classes}

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
            key, source = self.settings.api_key(cls.id, cls.env_vars)
            info = cls.describe()
            info["configured"] = bool(key)
            info["key_source"] = source
            info["key_hint"] = f"…{key[-4:]}" if key and len(key) > 8 else None
            result.append(info)
        return result

    def get(self, provider_id: str, capability: str | None = None) -> Provider:
        cls = self._class(provider_id)
        if capability and capability not in cls.capabilities:
            raise ProviderError(f"{cls.name} {MISSING_CAPABILITY[capability]}.", 400)
        key, _ = self.settings.api_key(cls.id, cls.env_vars)
        if not key:
            raise ProviderError(
                f"Für {cls.name} ist noch kein API-Schlüssel hinterlegt. Bitte unter ⚙ Einstellungen eintragen.", 400
            )
        return cls(key, self.client)
