"""Bildausschnitte für KI-Anbieter vorbereiten und Ergebnisse nahtlos zurückmischen.

Statt des ganzen (oft sehr großen) Bildes wird nur ein Ausschnitt um die Auswahl
mit etwas Umgebung gesendet. So arbeitet das Modell mit voller Auflösung am
gewünschten Bereich, und außerhalb der Auswahl bleibt jedes Pixel unverändert.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter

from ..engine import Change, NewImage, blend_into
from ..ops import OperationError
from ..selection import Box, Selection, mask_for_selection, render_mask
from .base import Provider

CONTEXT_MARGIN = 0.35  # Anteil der Auswahlgröße, der als Umgebung mitgesendet wird
ENHANCE_MARGIN = 0.12  # beim KI-Schärfen wenig Umgebung, damit die Auswahl viele Pixel bekommt

ENHANCE_PROMPT = (
    "Restore and sharpen this photo: remove blur, noise and compression artifacts and bring out fine, "
    "natural detail and crisp edges. Keep exactly the same content, composition, framing, colors, "
    "lighting and the identity of any people. Do not add, remove or move anything."
)


def _selection_area(image: Image.Image, selection: Selection | None) -> tuple[Box, Image.Image]:
    found = mask_for_selection(selection, image.width, image.height)
    if found is None:
        raise OperationError("Die Auswahl ist leer.")
    return found


def _expand(box: Box, margin: int, width: int, height: int) -> Box:
    return (max(0, box[0] - margin), max(0, box[1] - margin), min(width, box[2] + margin), min(height, box[3] + margin))


def fit_aspect(box: Box, aspect: float, width: int, height: int) -> Box:
    """Vergrößert ``box`` auf das Seitenverhältnis ``aspect`` (innerhalb des Bildes, ``box`` bleibt enthalten)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    w, h = min(width, round(w)), min(height, round(h))
    w, h = max(w, x1 - x0), max(h, y1 - y0)
    nx = round((x0 + x1) / 2 - w / 2)
    ny = round((y0 + y1) / 2 - h / 2)
    nx = min(max(0, nx), width - w)
    ny = min(max(0, ny), height - h)
    nx, ny = min(nx, x0), min(ny, y0)
    return (nx, ny, nx + w, ny + h)


def match_colors(result: Image.Image, reference: Image.Image, keep: np.ndarray) -> Image.Image:
    """Gleicht leichte Farb-/Helligkeitsabweichungen des KI-Ergebnisses an das Original an.

    Gemessen wird im unveränderten Umfeld der Auswahl. Hat das Modell das Umfeld gar nicht
    beibehalten (geringe Korrelation), wird nichts angeglichen.
    """
    if keep.sum() < 500:
        return result
    res = np.asarray(result, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    res_keep, ref_keep = res[keep].mean(axis=1), ref[keep].mean(axis=1)
    if res_keep.std() < 1 or ref_keep.std() < 1 or np.corrcoef(res_keep, ref_keep)[0, 1] < 0.6:
        return result
    out = res.copy()
    for c in range(3):
        r, o = res[..., c][keep], ref[..., c][keep]
        gain = float(np.clip(o.std() / max(r.std(), 1e-3), 0.9, 1.1))
        offset = float(np.clip(o.mean() - r.mean() * gain, -30, 30))
        out[..., c] = res[..., c] * gain + offset
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), "RGB")


def _with_alpha(processed: Image.Image, image: Image.Image, box: Box) -> Image.Image:
    if image.mode == "RGBA":
        processed = processed.convert("RGB")
        processed.putalpha(image.crop(box).getchannel("A"))
    return processed


def _fit_limits(img: Image.Image, limits: tuple[int, int, int]) -> Image.Image:
    min_side, max_side, max_pixels = limits
    w, h = img.size
    scale = 1.0
    if w * h > max_pixels:
        scale = math.sqrt(max_pixels / (w * h))
    if max(w, h) * scale > max_side:
        scale = max_side / max(w, h)
    if min(w, h) * scale < min_side:
        scale = min_side / min(w, h)
    if abs(scale - 1.0) < 1e-6:
        return img
    return img.resize((max(1, math.floor(w * scale)), max(1, math.floor(h * scale))), Image.Resampling.LANCZOS)


def _too_large(provider: Provider, width: int, height: int, max_side: int) -> str:
    return (
        f"Der Bereich ist für KI-Schärfen mit {provider.name} zu groß ({width} × {height} Pixel). "
        f"Bitte einen kleineren Ausschnitt wählen (längste Seite höchstens etwa {max_side} Pixel) – "
        "sonst würde das Ergebnis unschärfer statt schärfer. Tipp: „Als neues, vergrößertes Bild öffnen“ geht immer."
    )


def ai_edit(
    provider: Provider,
    image: Image.Image,
    selection: Selection | None,
    prompt: str,
    model: str | None,
    context_margin: float = CONTEXT_MARGIN,
) -> Change:
    """Bearbeitet die Auswahl per Text-Anweisung und mischt das Ergebnis weich ein."""
    width, height = image.size
    sbox, smask = _selection_area(image, selection)
    margin = max(16, round(context_margin * max(sbox[2] - sbox[0], sbox[3] - sbox[1])))
    context = _expand(sbox, margin, width, height) if selection is not None else sbox
    target_w, target_h = provider.edit_size((context[2] - context[0]) / (context[3] - context[1]), model)
    cbox = fit_aspect(context, target_w / target_h, width, height)

    crop = image.crop(cbox).convert("RGB")
    sent = crop.resize((target_w, target_h), Image.Resampling.LANCZOS)
    if selection is None:
        crop_mask = Image.new("L", crop.size, 255)
    else:
        crop_mask = render_mask(selection, width, height, cbox)
    provider_mask = crop_mask.point(lambda v: 255 if v >= 8 else 0).resize((target_w, target_h), Image.Resampling.NEAREST)
    provider_mask = provider_mask.filter(ImageFilter.MaxFilter(5))

    result = provider.edit(sent, provider_mask, prompt, model).convert("RGB")
    result = result.resize(crop.size, Image.Resampling.LANCZOS)
    result = match_colors(result, crop, np.asarray(crop_mask) == 0)

    rel = (sbox[0] - cbox[0], sbox[1] - cbox[1], sbox[2] - cbox[0], sbox[3] - cbox[1])
    processed = _with_alpha(result.crop(rel), image, sbox)
    return blend_into(image, processed, sbox, smask)


def ai_enhance(
    provider: Provider,
    image: Image.Image,
    selection: Selection | None,
    model: str | None,
    upscale_model: str | None = None,
    as_new: bool = False,
) -> Change | NewImage:
    """KI-Schärfen: im Bild (gleiche Größe) oder als vergrößertes neues Bild.

    Im Bild wird nur geschärft, wenn das Modell genug Pixel liefert – sonst gibt es eine
    verständliche Fehlermeldung statt eines unschärferen Ergebnisses.
    """
    width, height = image.size
    sbox, smask = _selection_area(image, selection)
    sw, sh = sbox[2] - sbox[0], sbox[3] - sbox[1]

    if "upscale" in provider.capabilities:
        work = _expand(sbox, 8, width, height)
        crop = image.crop(work).convert("RGB")
        prepared = _fit_limits(crop, provider.upscale_limits(upscale_model))
        if not as_new and prepared.width * 4 < crop.width:
            raise OperationError(_too_large(provider, sw, sh, 4 * max(prepared.size)))
        upscaled = provider.upscale(prepared, upscale_model).convert("RGB")
        rel = (sbox[0] - work[0], sbox[1] - work[1], sbox[2] - work[0], sbox[3] - work[1])
        if as_new:
            sx, sy = upscaled.width / crop.width, upscaled.height / crop.height
            scaled = (round(rel[0] * sx), round(rel[1] * sy), round(rel[2] * sx), round(rel[3] * sy))
            return NewImage(upscaled.crop(scaled), "KI-Detail-Zoom")
        if upscaled.width < crop.width:
            raise OperationError(f"{provider.name} hat ein kleineres Bild geliefert als erwartet – bitte einen kleineren Bereich wählen.")
        restored = upscaled.resize(crop.size, Image.Resampling.LANCZOS)
        return blend_into(image, _with_alpha(restored.crop(rel), image, sbox), sbox, smask)

    if "edit" not in provider.capabilities:
        raise OperationError(f"{provider.name} unterstützt kein KI-Schärfen.")
    target_w, target_h = provider.edit_size(sw / sh, model)
    if not as_new:
        # Das Modell liefert nur etwa target_w × target_h Pixel – größere Bereiche würden unschärfer.
        if sw * sh > target_w * target_h:
            raise OperationError(_too_large(provider, sw, sh, max(target_w, target_h)))
        return ai_edit(provider, image, selection, ENHANCE_PROMPT, model, context_margin=ENHANCE_MARGIN)

    cbox = fit_aspect(sbox, target_w / target_h, width, height)
    crop = image.crop(cbox).convert("RGB")
    sent = crop.resize((target_w, target_h), Image.Resampling.LANCZOS)
    result = provider.edit(sent, Image.new("L", sent.size, 255), ENHANCE_PROMPT, model).convert("RGB")
    sx, sy = result.width / crop.width, result.height / crop.height
    rel = (sbox[0] - cbox[0], sbox[1] - cbox[1], sbox[2] - cbox[0], sbox[3] - cbox[1])
    scaled = (round(rel[0] * sx), round(rel[1] * sy), round(rel[2] * sx), round(rel[3] * sy))
    return NewImage(result.crop(scaled), "KI-Detail-Zoom")
