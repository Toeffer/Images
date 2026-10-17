"""Gemeinsame Grundlage aller KI-Anbieter."""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

import httpx
from PIL import Image

from ..imageio import ImageLoadError, decode_image


class ProviderError(Exception):
    """Fehler eines KI-Anbieters; die Meldung ist für Benutzer gedacht."""

    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ModelOption:
    id: str
    label: str
    description: str = ""

    def to_json(self) -> dict[str, str]:
        return {"id": self.id, "label": self.label, "description": self.description}


@dataclass
class GenerateOptions:
    prompt: str
    model: str | None = None
    aspect_ratio: str = "1:1"
    quality: str | None = None
    resolution: str | None = None
    negative_prompt: str | None = None
    seed: int | None = None
    count: int = 1
    extra: dict[str, Any] = field(default_factory=dict)  # anbieterspezifisch, z. B. Schritte bei ComfyUI


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def b64_image(data: str) -> Image.Image:
    try:
        raw = base64.b64decode(data)
    except (ValueError, TypeError) as exc:
        raise ProviderError("Die Antwort des KI-Anbieters enthielt kein lesbares Bild.") from exc
    try:
        return decode_image(raw)
    except ImageLoadError as exc:
        raise ProviderError(str(exc)) from exc


def parse_ratio(ratio: str) -> float:
    w, _, h = ratio.partition(":")
    return float(w) / float(h)


def closest_ratio(aspect: float, ratios: tuple[str, ...]) -> str:
    return min(ratios, key=lambda r: abs(math.log(parse_ratio(r) / aspect)))


class Provider:
    """Basisklasse. Unterklassen beschreiben sich über Klassenattribute."""

    id: ClassVar[str]
    name: ClassVar[str]
    env_vars: ClassVar[tuple[str, ...]]
    key_url: ClassVar[str]
    capabilities: ClassVar[frozenset[str]]  # "generate", "edit", "upscale"
    models: ClassVar[tuple[ModelOption, ...]]
    default_model: ClassVar[str]
    aspect_ratios: ClassVar[tuple[str, ...]] = ("1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16")
    qualities: ClassVar[tuple[tuple[str, str], ...]] = ()
    resolutions: ClassVar[tuple[tuple[str, str], ...]] = ()
    upscale_models: ClassVar[tuple[ModelOption, ...]] = ()
    supports_negative_prompt: ClassVar[bool] = False
    supports_seed: ClassVar[bool] = False
    max_count: ClassVar[int] = 4
    timeout: ClassVar[float] = 240.0
    note: ClassVar[str] = ""
    requires_key: ClassVar[bool] = True  # False: lokaler Anbieter ohne API-Schlüssel
    local: ClassVar[bool] = False  # rechnet auf diesem Computer
    enhance_via_edit: ClassVar[bool] = True  # KI-Schärfen über eine Bearbeitungs-Anweisung erlaubt

    def __init__(self, api_key: str | None, client: httpx.Client, **context: Any) -> None:
        self.api_key = api_key
        self.client = client
        self.context = context  # gemeinsame Ressourcen der Registry (Einstellungen, lokale Modelle …)

    @classmethod
    def availability(cls, context: dict[str, Any]) -> dict[str, Any]:
        """Zustand eines Anbieters ohne API-Schlüssel; darf Felder von :meth:`describe` überschreiben."""
        return {"configured": True, "status": ""}

    # -- Schnittstelle -----------------------------------------------------------------
    def generate(self, opts: GenerateOptions) -> list[Image.Image]:
        raise ProviderError(f"{self.name} kann keine Bilder generieren.", 400)

    def edit(
        self, image: Image.Image, mask: Image.Image, prompt: str, model: str | None, options: dict[str, Any] | None = None
    ) -> Image.Image:
        """Bearbeitet ``image`` (RGB). ``mask`` (Modus L): 255 = darf verändert werden."""
        raise ProviderError(f"{self.name} kann keine Bilder bearbeiten.", 400)

    def upscale(self, image: Image.Image, model: str | None) -> Image.Image:
        raise ProviderError(f"{self.name} bietet kein KI-Hochskalieren an.", 400)

    def edit_size(self, aspect: float, model: str | None) -> tuple[int, int]:
        """Bildgröße, in der ein Ausschnitt zur Bearbeitung gesendet wird."""
        raise NotImplementedError

    def upscale_limits(self, model: str | None) -> tuple[int, int, int]:
        """(minimale Seitenlänge, maximale Seitenlänge, maximale Pixelzahl) für Hochskalieren."""
        return (64, 2048, 4_000_000)

    def model_or_default(self, model: str | None) -> str:
        return model or self.default_model

    # -- Beschreibung für die Oberfläche --------------------------------------------------
    @classmethod
    def describe(cls) -> dict[str, Any]:
        return {
            "id": cls.id,
            "name": cls.name,
            "capabilities": [c for c in ("generate", "edit", "upscale") if c in cls.capabilities],
            "enhance": "upscale" in cls.capabilities or ("edit" in cls.capabilities and cls.enhance_via_edit),
            "requires_key": cls.requires_key,
            "local": cls.local,
            "models": [m.to_json() for m in cls.models],
            "default_model": cls.default_model,
            "aspect_ratios": list(cls.aspect_ratios),
            "qualities": [{"value": v, "label": label} for v, label in cls.qualities],
            "resolutions": [{"value": v, "label": label} for v, label in cls.resolutions],
            "upscale_models": [m.to_json() for m in cls.upscale_models],
            "supports_negative_prompt": cls.supports_negative_prompt,
            "supports_seed": cls.supports_seed,
            "max_count": cls.max_count,
            "env_vars": list(cls.env_vars),
            "key_url": cls.key_url,
            "note": cls.note,
        }

    # -- HTTP ---------------------------------------------------------------------------
    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self.client.post(url, timeout=self.timeout, **kwargs)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"{self.name} hat nicht rechtzeitig geantwortet. Bitte erneut versuchen.", 504) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Keine Verbindung zu {self.name}: {exc}", 502) from exc
        if response.status_code >= 400:
            raise ProviderError(self._error_text(response), self._error_status(response.status_code))
        return response

    def json_of(self, response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.name} hat eine unerwartete Antwort geschickt.") from exc

    def extract_error(self, payload: Any) -> str | None:
        """Fehlermeldung aus der JSON-Antwort ziehen (anbieterspezifisch überschreibbar)."""
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])
            if isinstance(err, str):
                return err
            if payload.get("message"):
                return str(payload["message"])
        return None

    def _error_text(self, response: httpx.Response) -> str:
        try:
            detail = self.extract_error(response.json())
        except ValueError:
            detail = None
        if not detail:
            detail = response.text.strip()[:300] or f"HTTP {response.status_code}"
        status = response.status_code
        if status in (401, 403):
            reason = "API-Schlüssel ungültig oder ohne Berechtigung"
        elif status == 429:
            reason = "Zu viele Anfragen oder Guthaben aufgebraucht"
        elif status == 402:
            reason = "Guthaben aufgebraucht"
        elif status >= 500:
            reason = "Störung beim Anbieter – bitte später erneut versuchen"
        else:
            reason = "Anfrage abgelehnt"
        return f"{self.name}: {reason}. ({detail})"

    @staticmethod
    def _error_status(status: int) -> int:
        if status in (400, 401, 402, 403, 404, 413, 422, 429):
            return status if status != 401 else 403
        return 502
