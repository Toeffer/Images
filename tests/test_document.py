import numpy as np
import pytest
from PIL import Image

from bildwerk.document import Document, VersionConflict
from bildwerk.engine import Change

from .conftest import noise_image


def patch_change(box, color=(255, 0, 0)):
    return Change("patch", box=box, patch=Image.new("RGB", (box[2] - box[0], box[3] - box[1]), color))


def pixels(doc):
    return np.asarray(doc.image).copy()


def test_patch_undo_redo_roundtrip():
    doc = Document(noise_image(50, 40), "a.png")
    original = pixels(doc)
    info = doc.apply(patch_change((5, 5, 20, 15)), "Rot")
    assert info.kind == "patch" and info.box == (5, 5, 20, 15)
    changed = pixels(doc)
    assert (changed[5:15, 5:20] == [255, 0, 0]).all()
    assert doc.undo().box == (5, 5, 20, 15)
    assert np.array_equal(pixels(doc), original)
    doc.redo()
    assert np.array_equal(pixels(doc), changed)
    assert doc.info()["history"] == {"labels": ["Geöffnet", "Rot"], "position": 1}


def test_replace_with_size_change_and_goto():
    doc = Document(noise_image(50, 40), "a.png")
    doc.apply(patch_change((0, 0, 10, 10)), "A")
    doc.apply(Change.replace(Image.new("RGB", (20, 20))), "B")
    doc.apply(patch_change((2, 2, 4, 4)), "C")
    assert doc.image.size == (20, 20)
    info = doc.goto(1)
    assert info.kind == "replace" and doc.image.size == (50, 40) and doc.position == 1
    info = doc.goto(3)
    assert info.kind == "replace" and doc.position == 3


def test_goto_merges_patch_boxes():
    doc = Document(noise_image(50, 40), "a.png")
    doc.apply(patch_change((0, 0, 10, 10)), "A")
    doc.apply(patch_change((30, 20, 40, 30)), "B")
    info = doc.goto(0)
    assert info.kind == "patch" and info.box == (0, 0, 40, 30)


def test_new_edit_discards_redo_branch():
    doc = Document(noise_image(50, 40), "a.png")
    doc.apply(patch_change((0, 0, 10, 10)), "A")
    doc.apply(patch_change((0, 0, 10, 10)), "B")
    doc.undo()
    doc.apply(patch_change((0, 0, 5, 5)), "C")
    assert doc.info()["history"]["labels"] == ["Geöffnet", "A", "C"]
    assert doc.redo().kind == "none"


def test_history_is_trimmed_but_last_step_stays_undoable():
    doc = Document(noise_image(100, 100), "a.png", history_limit=100 * 100 * 3 * 3)
    for i in range(5):
        doc.apply(Change.replace(noise_image(100, 100, seed=i + 1)), f"Schritt {i}")
    labels = doc.info()["history"]["labels"]
    assert labels[0] == "Ältere Schritte entfernt"
    assert labels[-1] == "Schritt 4" and len(labels) < 6
    assert doc.undo().kind == "replace"


def test_pending_preview_lifecycle():
    doc = Document(noise_image(50, 40), "a.png")
    assert doc.set_pending(patch_change((0, 0, 5, 5)), "Vorschau", version=doc.version + 1) is None
    pending = doc.set_pending(patch_change((0, 0, 5, 5)), "Vorschau", version=doc.version, key="k")
    assert doc.take_matching_pending("k") is pending
    assert doc.take_matching_pending("other") is None
    doc.commit_pending(pending.id)
    assert doc.pending is None and doc.position == 1
    with pytest.raises(VersionConflict):
        doc.commit_pending(pending.id)


def test_pending_becomes_invalid_after_other_change():
    doc = Document(noise_image(50, 40), "a.png")
    pending = doc.set_pending(patch_change((0, 0, 5, 5)), "Vorschau", version=doc.version)
    doc.apply(patch_change((10, 10, 20, 20)), "Andere Änderung")
    with pytest.raises(VersionConflict):
        doc.commit_pending(pending.id)


def test_dirty_flag():
    doc = Document(noise_image(10, 10), "a.png")
    assert doc.info()["dirty"] is False
    doc.apply(patch_change((0, 0, 2, 2)), "A")
    assert doc.info()["dirty"] is True
    doc.mark_saved()
    assert doc.info()["dirty"] is False
