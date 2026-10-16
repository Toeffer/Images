"""Geometrische Operationen: Zuschneiden, Größe ändern, Drehen, Spiegeln, Detail-Zoom."""

from __future__ import annotations

from PIL import Image, ImageFilter

from ..imageio import normalize_mode
from . import OperationError, Param, operation
from .algorithms import rotated_crop_size

MAX_PIXELS = 100_000_000

RESAMPLE = {
    "lanczos": Image.Resampling.LANCZOS,
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
    "nearest": Image.Resampling.NEAREST,
}
RESAMPLE_CHOICES = (
    ("lanczos", "Lanczos (am schärfsten)"),
    ("bicubic", "Bikubisch"),
    ("bilinear", "Bilinear (weich)"),
    ("nearest", "Pixelgenau (Pixel-Art)"),
)


def _check_size(width: int, height: int) -> None:
    if width < 1 or height < 1:
        raise OperationError("Das Ergebnis wäre leer.")
    if width * height > MAX_PIXELS:
        raise OperationError(f"Das Ergebnis wäre mit {width} × {height} Pixeln zu groß (max. 100 Megapixel).")


@operation(
    "crop",
    "Zuschneiden",
    "transform",
    kind="transform",
    requires_selection=True,
    preview=False,
    description="Schneidet das Bild auf das Rechteck um die Auswahl zu.",
    shortcut="C",
)
def crop(img: Image.Image, ctx) -> Image.Image:
    if ctx.selection_box is None:
        raise OperationError("Zum Zuschneiden zuerst einen Bereich auswählen.")
    return img.crop(ctx.selection_box)


@operation(
    "resize",
    "Größe ändern",
    "transform",
    kind="transform",
    preview=False,
    params=[
        Param("mode", "Angabe", "choice", "percent", choices=(
            ("percent", "in Prozent"),
            ("width", "Breite (Höhe automatisch)"),
            ("height", "Höhe (Breite automatisch)"),
            ("exact", "Breite und Höhe"),
        )),
        Param("percent", "Skalierung", "float", 50, min=1, max=800, step=1, unit="%", show_if={"mode": ("percent",)}),
        Param("width", "Breite", "int", 0, min=1, max=30000, step=1, unit="px", default_from="width",
              show_if={"mode": ("width", "exact")}),
        Param("height", "Höhe", "int", 0, min=1, max=30000, step=1, unit="px", default_from="height",
              show_if={"mode": ("height", "exact")}),
        Param("resample", "Verfahren", "choice", "lanczos", choices=RESAMPLE_CHOICES),
    ],
    description="Skaliert das ganze Bild.",
)
def resize(img: Image.Image, ctx, *, mode: str, percent: float, width: int, height: int, resample: str) -> Image.Image:
    w, h = img.size
    if mode == "percent":
        new_w, new_h = round(w * percent / 100), round(h * percent / 100)
    elif mode == "width":
        new_w = width or w
        new_h = round(h * new_w / w)
    elif mode == "height":
        new_h = height or h
        new_w = round(w * new_h / h)
    else:
        new_w, new_h = width or w, height or h
    new_w, new_h = max(1, new_w), max(1, new_h)
    _check_size(new_w, new_h)
    return img.resize((new_w, new_h), RESAMPLE[resample], reducing_gap=3.0 if resample == "lanczos" else None)


@operation(
    "rotate90",
    "Drehen um 90°",
    "transform",
    kind="transform",
    preview=False,
    params=[
        Param("direction", "Richtung", "choice", "cw", choices=(
            ("cw", "90° im Uhrzeigersinn"),
            ("ccw", "90° gegen den Uhrzeigersinn"),
            ("180", "180°"),
        )),
    ],
    description="Dreht das Bild verlustfrei in 90°-Schritten.",
)
def rotate90(img: Image.Image, ctx, *, direction: str) -> Image.Image:
    method = {
        "cw": Image.Transpose.ROTATE_270,
        "ccw": Image.Transpose.ROTATE_90,
        "180": Image.Transpose.ROTATE_180,
    }[direction]
    return img.transpose(method)


@operation(
    "rotate",
    "Frei drehen / Begradigen",
    "transform",
    kind="transform",
    params=[
        Param("angle", "Winkel", "float", 0, min=-180, max=180, step=0.1, unit="°",
              help="Positive Werte drehen gegen den Uhrzeigersinn."),
        Param("crop", "Leere Ecken abschneiden", "bool", True,
              help="Schneidet das gedrehte Bild so zu, dass keine leeren Ecken entstehen."),
        Param("background", "Hintergrund", "choice", "white", choices=(
            ("white", "Weiß"),
            ("black", "Schwarz"),
            ("transparent", "Transparent"),
        ), show_if={"crop": (False,)}),
    ],
    description="Dreht um einen beliebigen Winkel, z. B. um einen schiefen Horizont gerade zu richten.",
)
def rotate(img: Image.Image, ctx, *, angle: float, crop: bool, background: str) -> Image.Image:
    if abs(angle) < 1e-6:
        return img.copy()
    source = img
    if not crop and background == "transparent" and source.mode != "RGBA":
        source = source.convert("RGBA")
    fill = {"white": "#ffffff", "black": "#000000", "transparent": (0, 0, 0, 0)}[background]
    if source.mode == "RGB" and background == "transparent":
        fill = "#ffffff"
    rotated = source.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=fill)
    if crop:
        cw, ch = rotated_crop_size(img.width, img.height, angle)
        left = (rotated.width - cw) // 2
        top = (rotated.height - ch) // 2
        rotated = rotated.crop((left, top, left + cw, top + ch))
    return normalize_mode(rotated)


@operation(
    "flip",
    "Spiegeln",
    "transform",
    kind="transform",
    preview=False,
    params=[
        Param("direction", "Richtung", "choice", "horizontal", choices=(
            ("horizontal", "Horizontal (links ↔ rechts)"),
            ("vertical", "Vertikal (oben ↕ unten)"),
        )),
    ],
)
def flip(img: Image.Image, ctx, *, direction: str) -> Image.Image:
    if direction == "horizontal":
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


@operation(
    "detail_zoom",
    "Detail-Zoom (Ausschnitt vergrößern)",
    "transform",
    kind="new_document",
    preview=False,
    params=[
        Param("factor", "Vergrößerung", "choice", "2", choices=(("2", "2×"), ("3", "3×"), ("4", "4×"), ("6", "6×"), ("8", "8×"))),
        Param("sharpen", "Nachschärfen", "int", 80, min=0, max=300, step=5, unit="%"),
        Param("resample", "Verfahren", "choice", "lanczos", choices=RESAMPLE_CHOICES[:2]),
    ],
    description="Vergrößert die Auswahl (oder das ganze Bild) und schärft nach. "
    "Das Ergebnis öffnet sich als neues Bild. Für KI-gestütztes Vergrößern siehe Reiter „KI“.",
    extra={"result_label": "Detail-Zoom"},
)
def detail_zoom(img: Image.Image, ctx, *, factor: str, sharpen: int, resample: str) -> Image.Image:
    region = img.crop(ctx.selection_box) if ctx.selection_box else img
    f = int(factor)
    new_w, new_h = region.width * f, region.height * f
    _check_size(new_w, new_h)
    out = region.resize((new_w, new_h), RESAMPLE[resample])
    if sharpen:
        radius = 0.6 + 0.35 * f
        if out.mode == "RGBA":
            alpha = out.getchannel("A")
            out = out.convert("RGB").filter(ImageFilter.UnsharpMask(radius=radius, percent=sharpen, threshold=1))
            out.putalpha(alpha)
        else:
            out = out.filter(ImageFilter.UnsharpMask(radius=radius, percent=sharpen, threshold=1))
    return out
