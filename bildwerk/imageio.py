"""Laden, Normalisieren und Exportieren von Bildern."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError, features

log = logging.getLogger(__name__)

# Sehr große Bilder bremsen den Browser aus; 100 Megapixel sind ein großzügiges Limit.
Image.MAX_IMAGE_PIXELS = 100_000_000

try:  # optionale HEIC-Unterstützung (iPhone-Fotos)
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:  # pragma: no cover - hängt von der Installation ab
    HEIC_SUPPORT = False


class ImageLoadError(ValueError):
    """Das Bild konnte nicht gelesen werden (Meldung ist für Benutzer gedacht)."""


@dataclass
class ImageMeta:
    """Metadaten, die beim Export wieder mitgeschrieben werden."""

    source_format: str | None = None
    icc_profile: bytes | None = None
    exif: bytes | None = None
    _display_transform: object | None = field(default=None, repr=False)
    _display_checked: bool = field(default=False, repr=False)


_ALL_EXPORT_FORMATS = {
    # id: (Pillow-Format, Dateiendung, MIME-Typ, unterstützt Transparenz, verlustbehaftet, Anzeigename)
    "png": ("PNG", "png", "image/png", True, False, "PNG"),
    "jpeg": ("JPEG", "jpg", "image/jpeg", False, True, "JPEG"),
    "webp": ("WEBP", "webp", "image/webp", True, True, "WebP"),
    "avif": ("AVIF", "avif", "image/avif", True, True, "AVIF"),
    "tiff": ("TIFF", "tif", "image/tiff", True, False, "TIFF"),
}
# Nur Formate anbieten, für die diese Pillow-Installation einen Encoder hat
EXPORT_FORMATS = {k: v for k, v in _ALL_EXPORT_FORMATS.items() if k not in ("webp", "avif") or features.check(k)}


def _to_8bit(im: Image.Image) -> Image.Image:
    """Wandelt 16-/32-Bit-Graustufen sinnvoll in 8 Bit um (statt sie abzuschneiden)."""
    arr = np.asarray(im).astype(np.float64)
    if im.mode.startswith("I;16"):
        arr = arr / 257.0
    elif arr.size and (arr.max() > 255 or arr.min() < 0):
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / max(hi - lo, 1e-9) * 255.0
    return Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8), "L")


def normalize_mode(im: Image.Image) -> Image.Image:
    """Bringt jedes Bild in den Modus RGB oder RGBA."""
    if im.mode in ("RGB", "RGBA"):
        return im
    if im.mode in ("LA", "PA", "RGBa", "La") or (im.mode == "P" and "transparency" in im.info):
        return im.convert("RGBA")
    if im.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        return _to_8bit(im).convert("RGB")
    return im.convert("RGB")


def load_image(data: bytes) -> tuple[Image.Image, ImageMeta]:
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Image.DecompressionBombError as exc:
        raise ImageLoadError("Das Bild ist zu groß (mehr als 100 Megapixel).") from exc
    except UnidentifiedImageError as exc:
        if data[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1", b"ftyphevc", b"ftypmsf1") and not HEIC_SUPPORT:
            raise ImageLoadError(
                "HEIC-Fotos (z. B. vom iPhone) brauchen ein Zusatzpaket: pip install pillow-heif – danach Bildwerk neu starten."
            ) from exc
        raise ImageLoadError("Dieses Dateiformat wird nicht unterstützt.") from exc
    except (OSError, SyntaxError, ValueError) as exc:
        raise ImageLoadError(f"Das Bild konnte nicht gelesen werden: {exc}") from exc

    fmt = im.format
    try:
        im = ImageOps.exif_transpose(im)  # Handyfotos richtig herum drehen
    except Exception:  # defekte EXIF-Daten sollen das Öffnen nicht verhindern
        log.warning("EXIF-Ausrichtung konnte nicht angewendet werden", exc_info=True)

    icc = im.info.get("icc_profile") or None
    if im.mode == "CMYK":
        im = _cmyk_to_rgb(im, icc)
        icc = None
    exif = None
    try:
        exif_obj = im.getexif()
        if len(exif_obj):
            exif = exif_obj.tobytes()
    except Exception:
        log.warning("EXIF-Daten konnten nicht gelesen werden", exc_info=True)

    return normalize_mode(im), ImageMeta(source_format=fmt, icc_profile=icc, exif=exif)


def _cmyk_to_rgb(im: Image.Image, icc: bytes | None) -> Image.Image:
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            return ImageCms.profileToProfile(im, src, ImageCms.createProfile("sRGB"), outputMode="RGB")
        except Exception:
            log.warning("CMYK-Profil konnte nicht angewendet werden", exc_info=True)
    return im.convert("RGB")


def _display_transform(meta: ImageMeta):
    """Farbtransformation vom eingebetteten Profil nach sRGB (nur für die Anzeige)."""
    if meta._display_checked:
        return meta._display_transform
    meta._display_checked = True
    if not meta.icc_profile:
        return None
    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(meta.icc_profile))
        description = (ImageCms.getProfileDescription(src) or "").lower()
        if "srgb" in description:
            return None
        perceptual = getattr(getattr(ImageCms, "Intent", None), "PERCEPTUAL", 0)
        meta._display_transform = ImageCms.buildTransform(
            src, ImageCms.createProfile("sRGB"), "RGB", "RGB", renderingIntent=perceptual
        )
    except Exception:
        log.warning("Farbprofil konnte nicht gelesen werden – Anzeige ohne Farbkorrektur", exc_info=True)
    return meta._display_transform


def to_display_rgba(im: Image.Image, meta: ImageMeta | None = None) -> bytes:
    """Rohdaten (RGBA, 8 Bit, sRGB) für die Anzeige im Browser-Canvas."""
    transform = _display_transform(meta) if meta else None
    if transform is not None:
        alpha = im.getchannel("A") if im.mode == "RGBA" else None
        rgb = ImageCms.applyTransform(im.convert("RGB"), transform)
        if alpha is not None:
            rgb.putalpha(alpha)
        im = rgb
    return im.convert("RGBA").tobytes()


def export_image(im: Image.Image, meta: ImageMeta | None, fmt: str, quality: int = 92) -> tuple[bytes, str, str]:
    """Kodiert das Bild. Rückgabe: (Daten, Dateiendung, MIME-Typ)."""
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"Unbekanntes Format: {fmt}")
    pil_format, ext, mime, alpha_ok, _lossy, _label = EXPORT_FORMATS[fmt]
    quality = max(1, min(100, int(quality)))

    out = im
    if out.mode == "RGBA" and not alpha_ok:
        background = Image.new("RGB", out.size, (255, 255, 255))
        background.paste(out, mask=out.getchannel("A"))
        out = background

    kwargs: dict = {}
    if meta and meta.icc_profile:
        kwargs["icc_profile"] = meta.icc_profile
    if meta and meta.exif and fmt in ("jpeg", "webp", "png", "avif"):
        kwargs["exif"] = meta.exif
    if fmt == "png":
        kwargs["optimize"] = False
        kwargs["compress_level"] = 6
    elif fmt == "jpeg":
        kwargs.update(quality=quality, optimize=True, subsampling=0 if quality >= 90 else 2)
    elif fmt in ("webp", "avif"):
        kwargs["quality"] = quality
    elif fmt == "tiff":
        kwargs["compression"] = "tiff_deflate"

    buf = io.BytesIO()
    try:
        out.save(buf, pil_format, **kwargs)
    except (OSError, KeyError, ValueError):
        # z. B. fehlende Encoder-Unterstützung für Metadaten – dann ohne Metadaten speichern
        log.warning("Speichern mit Metadaten fehlgeschlagen, versuche ohne", exc_info=True)
        kwargs.pop("exif", None)
        kwargs.pop("icc_profile", None)
        buf = io.BytesIO()
        out.save(buf, pil_format, **kwargs)
    return buf.getvalue(), ext, mime


def encode_png(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG", compress_level=1)
    return buf.getvalue()


def decode_image(data: bytes) -> Image.Image:
    """Dekodiert Bilddaten eines KI-Anbieters."""
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as exc:
        raise ImageLoadError("Der KI-Anbieter hat kein gültiges Bild geliefert.") from exc
    return normalize_mode(im)
