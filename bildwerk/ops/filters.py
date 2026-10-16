"""Schärfen, Weichzeichnen, Entrauschen und Effekte."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter

from . import Param, operation
from .algorithms import guided_filter, richardson_lucy


def _to_float(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) / 255.0


def _from_float(arr: np.ndarray, mode: str) -> Image.Image:
    return Image.fromarray(np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8), mode)


@operation(
    "sharpen",
    "Schärfen",
    "filter",
    params=[
        Param("amount", "Stärke", "int", 120, min=0, max=500, step=5, unit="%"),
        Param("radius", "Radius", "float", 1.5, min=0.1, max=20, step=0.1, unit="px",
              help="Kleine Werte betonen feine Details, große Werte gröbere Strukturen."),
        Param("threshold", "Schwellwert", "int", 2, min=0, max=50, step=1,
              help="Schützt ruhige Flächen (z. B. Haut, Himmel) vor Rauschverstärkung."),
    ],
    pad=lambda p: math.ceil(3 * p["radius"]) + 2,
    description="Klassisches „Unscharf maskieren“ – betont Kanten und Details.",
    shortcut="S",
)
def sharpen(img: Image.Image, ctx, *, amount: int, radius: float, threshold: int) -> Image.Image:
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=amount, threshold=threshold))


@operation(
    "deblur",
    "Unschärfe entfernen",
    "filter",
    params=[
        Param("sigma", "Unschärfe-Radius", "float", 1.5, min=0.3, max=6, step=0.1, unit="px",
              help="Wie stark das Bild ungefähr verwischt ist."),
        Param("iterations", "Durchläufe", "int", 20, min=1, max=60, step=1,
              help="Mehr Durchläufe holen mehr Details zurück."),
        Param("noise_protection", "Rauschschutz", "int", 65, min=0, max=100, step=1, unit="%",
              help="Verhindert, dass Bildrauschen mitverstärkt wird. Zu hohe Werte wirken glatt."),
        Param("strength", "Stärke", "int", 100, min=0, max=100, step=1, unit="%"),
    ],
    pad=lambda p: min(160, math.ceil(3 * p["sigma"] * math.sqrt(2 * p["iterations"])) + 8),
    description="Rekonstruiert leicht unscharfe Fotos (Richardson-Lucy-Dekonvolution). "
    "Am besten auf einen Bildausschnitt anwenden – bei großen Bildern dauert es einige Sekunden.",
)
def deblur(img: Image.Image, ctx, *, sigma: float, iterations: int, noise_protection: int, strength: int) -> Image.Image:
    y, cb, cr = img.convert("YCbCr").split()
    restored = richardson_lucy(_to_float(y), sigma, iterations, tv=0.006 * noise_protection / 100)
    out = Image.merge("YCbCr", (_from_float(restored, "L"), cb, cr)).convert("RGB")
    return Image.blend(img, out, strength / 100) if strength < 100 else out


@operation(
    "clarity",
    "Klarheit",
    "filter",
    params=[
        Param("amount", "Stärke", "int", 40, min=0, max=200, step=1, unit="%"),
        Param("radius", "Radius", "int", 30, min=5, max=150, step=1, unit="px"),
    ],
    pad=lambda p: 3 * p["radius"] + 2,
    description="Verstärkt den lokalen Kontrast – Strukturen wirken plastischer.",
)
def clarity(img: Image.Image, ctx, *, amount: int, radius: int) -> Image.Image:
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=amount, threshold=0))


@operation(
    "blur",
    "Weichzeichnen",
    "filter",
    params=[Param("radius", "Radius", "float", 4, min=0.1, max=100, step=0.1, unit="px")],
    pad=lambda p: math.ceil(3 * p["radius"]) + 2,
    description="Gaußscher Weichzeichner, z. B. für Hintergründe.",
)
def blur(img: Image.Image, ctx, *, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius))


@operation(
    "denoise",
    "Rauschen reduzieren",
    "filter",
    params=[
        Param("strength", "Stärke", "int", 40, min=1, max=100, step=1, unit="%"),
        Param("radius", "Radius", "int", 3, min=1, max=12, step=1, unit="px"),
    ],
    pad=lambda p: 2 * p["radius"] + 2,
    description="Glättet Bildrauschen und erhält dabei Kanten (Guided Filter).",
)
def denoise(img: Image.Image, ctx, *, strength: int, radius: int) -> Image.Image:
    eps = (0.004 + strength / 100 * 0.09) ** 2
    arr = _to_float(img)
    out = np.stack([guided_filter(arr[..., c], radius, eps) for c in range(arr.shape[2])], axis=-1)
    return _from_float(out, img.mode)


@operation(
    "pixelate",
    "Verpixeln",
    "filter",
    params=[Param("block", "Blockgröße", "int", 16, min=2, max=200, step=1, unit="px")],
    description="Macht Bereiche unkenntlich – z. B. Gesichter oder Kennzeichen.",
)
def pixelate(img: Image.Image, ctx, *, block: int) -> Image.Image:
    w, h = img.size
    cols, rows = math.ceil(w / block), math.ceil(h / block)
    arr = np.asarray(img, dtype=np.float32)
    arr = np.pad(arr, ((0, rows * block - h), (0, cols * block - w), (0, 0)), mode="edge")
    means = arr.reshape(rows, block, cols, block, arr.shape[2]).mean(axis=(1, 3))
    out = np.repeat(np.repeat(means, block, axis=0), block, axis=1)[:h, :w]
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), img.mode)


@operation(
    "fill",
    "Füllen",
    "paint",
    params=[
        Param("color", "Farbe", "color", "#ffffff"),
        Param("opacity", "Deckkraft", "int", 100, min=1, max=100, step=1, unit="%"),
    ],
    description="Füllt die Auswahl (oder das ganze Bild) mit einer Farbe.",
    alpha="process",
)
def fill(img: Image.Image, ctx, *, color: str, opacity: int) -> Image.Image:
    layer = Image.new(img.mode, img.size, color)  # bei RGBA deckend
    return Image.blend(img, layer, opacity / 100)


@operation(
    "erase",
    "Löschen (transparent)",
    "paint",
    description="Macht die Auswahl durchsichtig. Zum Speichern PNG, WebP oder TIFF verwenden.",
    requires_selection=True,
    alpha="process",
    needs_alpha=True,
)
def erase(img: Image.Image, ctx) -> Image.Image:
    out = img.copy()
    out.putalpha(0)
    return out
