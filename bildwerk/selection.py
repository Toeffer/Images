"""Auswahlen (Rechteck, Ellipse, Lasso, Pinsel) und ihre Umwandlung in Graustufen-Masken.

Eine Auswahl besteht aus einer Folge von Formen, die nacheinander hinzugefügt oder
abgezogen werden. Danach kann sie umgekehrt und mit einer weichen Kante versehen
werden. Alle Koordinaten sind Bildpixel (Gleitkomma, Ursprung oben links).
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pydantic import BaseModel, Field

Box = tuple[int, int, int, int]  # (x0, y0, x1, y1), x1/y1 exklusiv

Mode = Literal["add", "subtract"]
MAX_POINTS = 20_000


class RectShape(BaseModel):
    type: Literal["rect"] = "rect"
    x: float
    y: float
    w: float
    h: float
    mode: Mode = "add"


class EllipseShape(BaseModel):
    type: Literal["ellipse"] = "ellipse"
    x: float
    y: float
    w: float
    h: float
    mode: Mode = "add"


class PolygonShape(BaseModel):
    type: Literal["polygon"] = "polygon"
    points: list[tuple[float, float]] = Field(min_length=3, max_length=MAX_POINTS)
    mode: Mode = "add"


class BrushShape(BaseModel):
    type: Literal["brush"] = "brush"
    points: list[tuple[float, float]] = Field(min_length=1, max_length=MAX_POINTS)
    size: float = Field(gt=0, le=5000)  # Pinseldurchmesser in Pixeln
    mode: Mode = "add"


Shape = Annotated[Union[RectShape, EllipseShape, PolygonShape, BrushShape], Field(discriminator="type")]


class Selection(BaseModel):
    shapes: list[Shape] = Field(default_factory=list, max_length=1000)
    feather: float = Field(0, ge=0, le=500)  # weiche Kante in Pixeln
    invert: bool = False

    def is_empty(self) -> bool:
        return not self.invert and not any(s.mode == "add" for s in self.shapes)


def _shape_bounds(shape: Shape) -> tuple[float, float, float, float]:
    if isinstance(shape, (RectShape, EllipseShape)):
        x0, x1 = sorted((shape.x, shape.x + shape.w))
        y0, y1 = sorted((shape.y, shape.y + shape.h))
        return x0, y0, x1, y1
    xs = [p[0] for p in shape.points]
    ys = [p[1] for p in shape.points]
    r = shape.size / 2 if isinstance(shape, BrushShape) else 0.0
    return min(xs) - r, min(ys) - r, max(xs) + r, max(ys) + r


def _feather_margin(feather: float) -> int:
    # Gauß mit sigma = feather / 2 – nach 3 sigma ist der Einfluss vernachlässigbar.
    return math.ceil(1.5 * feather) + 2 if feather > 0 else 0


def _clip(box: tuple[float, float, float, float], width: int, height: int) -> Box | None:
    x0 = max(0, math.floor(box[0]))
    y0 = max(0, math.floor(box[1]))
    x1 = min(width, math.ceil(box[2]))
    y1 = min(height, math.ceil(box[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def selection_bounds(sel: Selection, width: int, height: int, *, include_feather: bool = True) -> Box | None:
    """Grobe Begrenzung des Bereichs, in dem die Maske nicht null sein kann (im Bild geclippt)."""
    if sel.invert:
        return (0, 0, width, height)
    boxes = [_shape_bounds(s) for s in sel.shapes if s.mode == "add"]
    if not boxes:
        return None
    m = _feather_margin(sel.feather) if include_feather else 0
    union = (
        min(b[0] for b in boxes) - m,
        min(b[1] for b in boxes) - m,
        max(b[2] for b in boxes) + m,
        max(b[3] for b in boxes) + m,
    )
    return _clip(union, width, height)


def _rasterize(shapes: list[Shape], region: Box) -> Image.Image:
    """Zeichnet die Formen kantengeglättet (per Supersampling) in den Bereich ``region``."""
    rx0, ry0, rx1, ry1 = region
    rw, rh = rx1 - rx0, ry1 - ry0
    area = rw * rh
    s = 4 if area <= 2_000_000 else 2 if area <= 8_000_000 else 1
    canvas = Image.new("L", (rw * s, rh * s), 0)
    draw = ImageDraw.Draw(canvas)

    def tx(x: float) -> float:
        return (x - rx0) * s

    def ty(y: float) -> float:
        return (y - ry0) * s

    for shape in shapes:
        fill = 255 if shape.mode == "add" else 0
        if isinstance(shape, (RectShape, EllipseShape)):
            x0, y0, x1, y1 = _shape_bounds(shape)
            box = [round(tx(x0)), round(ty(y0)), round(tx(x1)) - 1, round(ty(y1)) - 1]
            if box[2] < box[0] or box[3] < box[1]:
                continue
            if isinstance(shape, RectShape):
                draw.rectangle(box, fill=fill)
            else:
                draw.ellipse(box, fill=fill)
        elif isinstance(shape, PolygonShape):
            draw.polygon([(tx(x), ty(y)) for x, y in shape.points], fill=fill)
        else:
            width = max(1, round(shape.size * s))
            radius = shape.size * s / 2
            pts = [(tx(x), ty(y)) for x, y in shape.points]
            if len(pts) > 1:
                draw.line(pts, fill=fill, width=width, joint="curve")
            for px, py in pts:
                draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=fill)

    if s > 1:
        canvas = canvas.resize((rw, rh), Image.Resampling.BOX)
    return canvas


def render_mask(sel: Selection, width: int, height: int, box: Box | None = None) -> Image.Image:
    """Erzeugt die Auswahlmaske (Modus "L", 255 = ausgewählt) für den Ausschnitt ``box``."""
    if box is None:
        box = (0, 0, width, height)
    bx0, by0, bx1, by1 = box
    m = _feather_margin(sel.feather)
    region = (max(0, bx0 - m), max(0, by0 - m), min(width, bx1 + m), min(height, by1 + m))
    mask = _rasterize(list(sel.shapes), region)

    if sel.feather > 0:
        # Am Bildrand wird die Maske fortgesetzt, damit Auswahlen, die bis an den Rand
        # reichen, dort nicht ausblenden.
        padded = Image.fromarray(np.pad(np.asarray(mask), m, mode="edge"))
        blurred = padded.filter(ImageFilter.GaussianBlur(sel.feather / 2))
        mask = blurred.crop((m, m, m + mask.width, m + mask.height))

    if sel.invert:
        mask = Image.eval(mask, lambda v: 255 - v)

    ox, oy = bx0 - region[0], by0 - region[1]
    return mask.crop((ox, oy, ox + (bx1 - bx0), oy + (by1 - by0)))


def mask_for_selection(sel: Selection | None, width: int, height: int) -> tuple[Box, Image.Image] | None:
    """Liefert (enge Begrenzung, Maske in dieser Begrenzung) oder ``None`` bei leerer Auswahl.

    Ohne Auswahl (``sel is None``) gilt das ganze Bild als ausgewählt.
    """
    if sel is None:
        return (0, 0, width, height), Image.new("L", (width, height), 255)
    if sel.is_empty():
        return None
    bounds = selection_bounds(sel, width, height)
    if bounds is None:
        return None
    mask = render_mask(sel, width, height, bounds)
    tight = mask.getbbox()
    if tight is None:
        return None
    box = (bounds[0] + tight[0], bounds[1] + tight[1], bounds[0] + tight[2], bounds[1] + tight[3])
    return box, mask.crop(tight)


def shape_box(sel: Selection, width: int, height: int) -> Box | None:
    """Begrenzung der Auswahl ohne weiche Kante – z. B. für das Zuschneiden."""
    if sel.invert:
        return (0, 0, width, height)
    found = mask_for_selection(Selection(shapes=sel.shapes), width, height)
    return found[0] if found else None
