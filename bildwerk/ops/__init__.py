"""Register aller Bildoperationen.

Eine Operation ist eine Python-Funktion mit beschreibenden Parametern. Die Oberfläche
baut ihre Regler automatisch aus diesen Beschreibungen – eine neue Operation braucht
deshalb keine Änderung im Frontend. Beispiel::

    @operation("posterize", "Tontrennung", "filter",
               params=[Param("bits", "Bits pro Kanal", "int", 4, min=1, max=8)])
    def posterize(img, ctx, *, bits):
        return ImageOps.posterize(img, bits)

Arten von Operationen:

``filter``
    Arbeitet auf einem Bildausschnitt und liefert ein gleich großes Bild. Wirkt nur
    innerhalb der Auswahl (mit weicher Kante), ohne Auswahl auf das ganze Bild.
``transform``
    Erhält das ganze Bild und darf dessen Größe ändern (z. B. Drehen, Zuschneiden).
``new_document``
    Erzeugt aus dem Bild ein neues Dokument (z. B. Detail-Zoom).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from PIL import Image

from ..selection import Box


class OperationError(ValueError):
    """Fehler, dessen Meldung direkt in der Oberfläche angezeigt werden kann."""


@dataclass(frozen=True)
class OpContext:
    """Zusatzinformationen für eine Operation."""

    image_size: tuple[int, int]
    selection_box: Box | None = None  # Begrenzung der Auswahl ohne weiche Kante


ParamType = Literal["float", "int", "bool", "choice", "color"]
_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


@dataclass(frozen=True)
class Param:
    name: str
    label: str
    type: ParamType
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    unit: str = ""
    choices: tuple[tuple[str, str], ...] = ()
    help: str = ""
    default_from: str | None = None  # "width" / "height": Vorgabe aus der Bildgröße
    show_if: dict[str, tuple[Any, ...]] | None = None  # nur anzeigen, wenn andere Parameter passen

    def coerce(self, value: Any) -> Any:
        if value is None:
            return self.default
        if self.type in ("float", "int"):
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise OperationError(f"„{self.label}“ muss eine Zahl sein.") from None
            if not math.isfinite(number):
                return self.default
            if self.min is not None:
                number = max(self.min, number)
            if self.max is not None:
                number = min(self.max, number)
            return int(round(number)) if self.type == "int" else number
        if self.type == "bool":
            if isinstance(value, str):
                return value.lower() in ("1", "true", "on", "ja", "yes")
            return bool(value)
        if self.type == "choice":
            value = str(value)
            if value not in {c[0] for c in self.choices}:
                raise OperationError(f"Ungültige Auswahl für „{self.label}“: {value}")
            return value
        if self.type == "color":
            value = str(value)
            if not _COLOR_RE.match(value):
                raise OperationError(f"„{self.label}“ muss eine Farbe wie #ff8800 sein.")
            return value.lower()
        raise AssertionError(self.type)

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name, "label": self.label, "type": self.type, "default": self.default}
        for key in ("min", "max", "step", "default_from"):
            if getattr(self, key) is not None:
                data[key] = getattr(self, key)
        if self.unit:
            data["unit"] = self.unit
        if self.help:
            data["help"] = self.help
        if self.choices:
            data["choices"] = [{"value": v, "label": label} for v, label in self.choices]
        if self.show_if:
            data["show_if"] = {k: list(v) for k, v in self.show_if.items()}
        return data


OpKind = Literal["filter", "transform", "new_document"]


@dataclass(frozen=True)
class Operation:
    id: str
    label: str
    category: str
    func: Callable[..., Image.Image]
    kind: OpKind = "filter"
    params: tuple[Param, ...] = ()
    description: str = ""
    pad: int | Callable[[dict[str, Any]], int] = 0  # nötiger Rand um den Ausschnitt (Filterradius)
    requires_selection: bool = False
    preview: bool = True
    alpha: Literal["keep", "process"] = "keep"  # keep: Funktion sieht nur RGB, Alpha bleibt erhalten
    needs_alpha: bool = False  # Bild wird vorher in RGBA umgewandelt (z. B. Löschen)
    shortcut: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self, raw: dict[str, Any] | None) -> dict[str, Any]:
        raw = raw or {}
        return {p.name: p.coerce(raw.get(p.name)) for p in self.params}

    def padding(self, params: dict[str, Any]) -> int:
        return self.pad(params) if callable(self.pad) else int(self.pad)

    def run(self, img: Image.Image, ctx: OpContext, params: dict[str, Any]) -> Image.Image:
        if self.kind == "filter" and self.alpha == "keep" and img.mode == "RGBA":
            alpha = img.getchannel("A")
            out = self.func(img.convert("RGB"), ctx, **params).convert("RGB")
            out.putalpha(alpha)
            return out
        out = self.func(img, ctx, **params)
        if self.kind == "filter" and out.mode != img.mode:
            out = out.convert(img.mode)
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "category": self.category,
            "kind": self.kind,
            "description": self.description,
            "params": [p.to_json() for p in self.params],
            "requires_selection": self.requires_selection,
            "preview": self.preview,
            "shortcut": self.shortcut,
        }


CATEGORIES: dict[str, str] = {
    "adjust": "Anpassungen",
    "filter": "Filter & Schärfe",
    "paint": "Füllen & Löschen",
    "transform": "Transformieren",
}

REGISTRY: dict[str, Operation] = {}


def operation(op_id: str, label: str, category: str, **kwargs: Any) -> Callable:
    """Dekorator zum Registrieren einer Operation (siehe Moduldokumentation)."""
    if category not in CATEGORIES:
        raise ValueError(f"Unbekannte Kategorie: {category}")

    def decorator(func: Callable[..., Image.Image]) -> Callable[..., Image.Image]:
        if op_id in REGISTRY:
            raise ValueError(f"Operation {op_id!r} ist bereits registriert")
        params = tuple(kwargs.pop("params", ()))
        REGISTRY[op_id] = Operation(id=op_id, label=label, category=category, func=func, params=params, **kwargs)
        return func

    return decorator


def get_operation(op_id: str) -> Operation:
    try:
        return REGISTRY[op_id]
    except KeyError:
        raise OperationError(f"Unbekannte Operation: {op_id}") from None


def catalog() -> list[dict[str, Any]]:
    return [
        {"id": cat_id, "label": cat_label, "operations": [op.to_json() for op in REGISTRY.values() if op.category == cat_id]}
        for cat_id, cat_label in CATEGORIES.items()
    ]


# Registrierung der mitgelieferten Operationen
from . import adjust, filters, transform  # noqa: E402,F401
