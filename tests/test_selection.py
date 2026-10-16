import numpy as np
import pytest

from bildwerk.selection import (
    BrushShape,
    EllipseShape,
    PolygonShape,
    RectShape,
    Selection,
    mask_for_selection,
    render_mask,
    shape_box,
)

W, H = 200, 120


def full_mask(sel):
    return np.asarray(render_mask(sel, W, H))


def test_rect_mask_is_pixel_exact():
    sel = Selection(shapes=[RectShape(x=10, y=20, w=30, h=40)])
    box, mask = mask_for_selection(sel, W, H)
    assert box == (10, 20, 40, 60)
    assert np.asarray(mask).min() == 255
    m = full_mask(sel)
    assert m[20:60, 10:40].min() == 255
    assert m.sum() == 255 * 30 * 40


def test_negative_width_rect_is_normalized():
    sel = Selection(shapes=[RectShape(x=40, y=60, w=-30, h=-40)])
    assert mask_for_selection(sel, W, H)[0] == (10, 20, 40, 60)


def test_ellipse_inside_and_outside():
    m = full_mask(Selection(shapes=[EllipseShape(x=0, y=0, w=200, h=120)]))
    assert m[60, 100] == 255
    assert m[2, 2] == 0


def test_polygon_and_brush():
    m = full_mask(Selection(shapes=[PolygonShape(points=[(10, 10), (110, 10), (10, 110)])]))
    assert m[20, 20] == 255 and m[100, 100] == 0
    brush = Selection(shapes=[BrushShape(points=[(50, 60), (150, 60)], size=20)])
    box, mask = mask_for_selection(brush, W, H)
    assert box == (40, 50, 160, 70)
    m = full_mask(brush)
    assert m[60, 100] == 255 and m[60, 45] == 255 and m[80, 100] == 0


def test_subtract_cuts_a_hole():
    sel = Selection(shapes=[RectShape(x=0, y=0, w=W, h=H), RectShape(x=50, y=30, w=40, h=40, mode="subtract")])
    m = full_mask(sel)
    assert m[50, 70] == 0 and m[10, 10] == 255


def test_invert_flag():
    sel = Selection(shapes=[RectShape(x=0, y=0, w=100, h=H)], invert=True)
    box, mask = mask_for_selection(sel, W, H)
    assert box == (100, 0, 200, 120)
    assert np.asarray(mask).min() == 255


def test_feather_creates_soft_edge():
    sel = Selection(shapes=[RectShape(x=60, y=30, w=80, h=60)], feather=10)
    box, mask = mask_for_selection(sel, W, H)
    assert box[0] < 60 and box[2] > 140  # Rand wird weich nach außen erweitert
    m = full_mask(sel)
    assert m[60, 100] == 255  # Mitte voll ausgewählt
    assert 40 < m[60, 60] < 215  # an der Kante etwa halb
    assert m[60, 2] == 0


def test_feather_does_not_fade_at_image_border():
    sel = Selection(shapes=[RectShape(x=0, y=0, w=W, h=H)], feather=20)
    assert full_mask(sel).min() == 255


@pytest.mark.parametrize("shapes", [[], [RectShape(x=0, y=0, w=10, h=10, mode="subtract")], [RectShape(x=500, y=500, w=10, h=10)]])
def test_empty_selection(shapes):
    assert mask_for_selection(Selection(shapes=shapes), W, H) is None


def test_no_selection_means_whole_image():
    box, mask = mask_for_selection(None, W, H)
    assert box == (0, 0, W, H) and np.asarray(mask).min() == 255


def test_shape_box_ignores_feather():
    sel = Selection(shapes=[RectShape(x=60, y=30, w=80, h=60)], feather=25)
    assert shape_box(sel, W, H) == (60, 30, 140, 90)
