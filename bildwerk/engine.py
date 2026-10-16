"""Führt Operationen aus – auf Wunsch nur innerhalb einer Auswahl."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from PIL import Image

from .ops import Operation, OperationError, OpContext
from .selection import Box, Selection, mask_for_selection, shape_box


@dataclass
class Change:
    """Ergebnis einer Bearbeitung: neues Teilstück (``patch``) oder neues Gesamtbild (``replace``)."""

    kind: Literal["patch", "replace"]
    box: Box | None = None
    patch: Image.Image | None = None
    image: Image.Image | None = None

    @classmethod
    def replace(cls, image: Image.Image) -> "Change":
        return cls("replace", image=image)

    @property
    def result(self) -> Image.Image:
        return self.patch if self.kind == "patch" else self.image


@dataclass
class NewImage:
    """Ergebnis einer Operation, die ein neues Dokument erzeugt."""

    image: Image.Image
    label: str


def blend_into(image: Image.Image, processed: Image.Image, box: Box, mask: Image.Image) -> Change:
    """Mischt ``processed`` (Größe von ``box``) über die Maske in ``image`` ein."""
    original = image.crop(box)
    if processed.mode != original.mode:
        processed = processed.convert(original.mode)
    patch = Image.composite(processed, original, mask)
    if box == (0, 0, image.width, image.height):
        return Change.replace(patch)
    return Change("patch", box=box, patch=patch)


def run_operation(
    op: Operation, image: Image.Image, raw_params: dict[str, Any] | None, selection: Selection | None
) -> Change | NewImage:
    params = op.validate(raw_params)
    width, height = image.size

    if op.kind in ("transform", "new_document"):
        sel_box = shape_box(selection, width, height) if selection is not None else None
        if op.requires_selection and sel_box is None:
            raise OperationError("Bitte zuerst einen Bereich auswählen.")
        result = op.run(image, OpContext((width, height), sel_box), params)
        if op.kind == "new_document":
            return NewImage(result, op.extra.get("result_label", op.label))
        return Change.replace(result)

    work_image = image
    force_replace = False
    if op.needs_alpha and image.mode != "RGBA":
        work_image = image.convert("RGBA")
        force_replace = True

    ctx = OpContext((width, height))
    if selection is None:
        if op.requires_selection:
            raise OperationError("Bitte zuerst einen Bereich auswählen.")
        return Change.replace(op.run(work_image, ctx, params))

    found = mask_for_selection(selection, width, height)
    if found is None:
        raise OperationError("Die Auswahl ist leer.")
    box, mask = found
    pad = op.padding(params)
    work = (max(0, box[0] - pad), max(0, box[1] - pad), min(width, box[2] + pad), min(height, box[3] + pad))
    processed = op.run(work_image.crop(work), ctx, params)
    inner = (box[0] - work[0], box[1] - work[1], box[2] - work[0], box[3] - work[1])
    change = blend_into(work_image, processed.crop(inner), box, mask)

    if force_replace and change.kind == "patch":
        full = work_image.copy()
        full.paste(change.patch, box[:2])
        return Change.replace(full)
    return change
