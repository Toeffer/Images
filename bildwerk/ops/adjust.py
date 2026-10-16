"""Tonwert- und Farbkorrekturen."""

from __future__ import annotations

from PIL import Image, ImageEnhance, ImageOps

from . import Param, operation


def _clip(v: float) -> int:
    return 0 if v < 0 else 255 if v > 255 else int(round(v))


def _apply_lut(img: Image.Image, lut: list[int]) -> Image.Image:
    return img.point(lut * len(img.getbands()))


@operation(
    "brightness_contrast",
    "Helligkeit & Kontrast",
    "adjust",
    params=[
        Param("brightness", "Helligkeit", "int", 0, min=-100, max=100, step=1),
        Param("contrast", "Kontrast", "int", 0, min=-100, max=100, step=1),
    ],
    description="Hellt auf oder dunkelt ab und verstärkt bzw. verringert den Kontrast.",
)
def brightness_contrast(img: Image.Image, ctx, *, brightness: int, contrast: int) -> Image.Image:
    c = contrast * 2.55
    factor = (259 * (c + 255)) / (255 * (259 - c))
    offset = brightness * 1.28
    return _apply_lut(img, [_clip(factor * (v - 128) + 128 + offset) for v in range(256)])


@operation(
    "levels",
    "Tonwerte",
    "adjust",
    params=[
        Param("black", "Schwarzpunkt", "int", 0, min=0, max=254, step=1),
        Param("white", "Weißpunkt", "int", 255, min=1, max=255, step=1),
        Param("gamma", "Mitteltöne (Gamma)", "float", 1.0, min=0.1, max=4.0, step=0.01,
              help="Werte über 1 hellen die Mitteltöne auf, Werte unter 1 dunkeln sie ab."),
    ],
    description="Legt fest, welche Helligkeit reines Schwarz bzw. Weiß wird.",
)
def levels(img: Image.Image, ctx, *, black: int, white: int, gamma: float) -> Image.Image:
    white = max(white, black + 1)
    span = white - black
    lut = []
    for v in range(256):
        x = min(1.0, max(0.0, (v - black) / span))
        lut.append(_clip((x ** (1.0 / gamma)) * 255))
    return _apply_lut(img, lut)


@operation(
    "color",
    "Farbe",
    "adjust",
    params=[
        Param("saturation", "Sättigung", "int", 0, min=-100, max=100, step=1),
        Param("temperature", "Farbtemperatur", "int", 0, min=-100, max=100, step=1,
              help="Negativ = kühler (blauer), positiv = wärmer (gelber)."),
        Param("tint", "Tönung", "int", 0, min=-100, max=100, step=1,
              help="Negativ = grüner, positiv = magentafarbener."),
    ],
    description="Sättigung, Farbtemperatur und Tönung anpassen.",
)
def color(img: Image.Image, ctx, *, saturation: int, temperature: int, tint: int) -> Image.Image:
    out = img
    if saturation:
        out = ImageEnhance.Color(out).enhance(1 + saturation / 100)
    if temperature or tint:
        t = temperature / 100 * 0.2
        g = tint / 100 * 0.15
        gains = (1 + t, 1 - g, 1 - t)
        r, gr, b = out.split()
        bands = [band.point([_clip(v * gain) for v in range(256)]) for band, gain in zip((r, gr, b), gains)]
        out = Image.merge("RGB", bands)
    return out


@operation(
    "autocontrast",
    "Auto-Kontrast",
    "adjust",
    params=[
        Param("cutoff", "Beschnitt", "float", 0.5, min=0, max=10, step=0.1, unit="%",
              help="Anteil der hellsten und dunkelsten Pixel, der ignoriert wird."),
    ],
    description="Spreizt den Tonwertumfang automatisch auf Schwarz bis Weiß.",
)
def autocontrast(img: Image.Image, ctx, *, cutoff: float) -> Image.Image:
    return ImageOps.autocontrast(img, cutoff=cutoff, preserve_tone=True)


@operation("grayscale", "Schwarzweiß", "adjust", description="Wandelt in Graustufen um.")
def grayscale(img: Image.Image, ctx) -> Image.Image:
    return ImageOps.grayscale(img).convert("RGB")


@operation(
    "sepia",
    "Sepia",
    "adjust",
    params=[Param("strength", "Stärke", "int", 100, min=0, max=100, step=1, unit="%")],
    description="Warmer Altfoto-Look.",
)
def sepia(img: Image.Image, ctx, *, strength: int) -> Image.Image:
    toned = ImageOps.colorize(ImageOps.grayscale(img), black=(26, 16, 8), white=(255, 246, 225), mid=(164, 124, 84))
    return Image.blend(img, toned, strength / 100)


@operation("invert", "Invertieren", "adjust", description="Erzeugt ein Negativ.")
def invert(img: Image.Image, ctx) -> Image.Image:
    return ImageOps.invert(img)
