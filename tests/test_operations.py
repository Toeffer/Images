import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from bildwerk.engine import Change, NewImage, run_operation
from bildwerk.ops import REGISTRY, OperationError, Param, catalog, get_operation
from bildwerk.ops.algorithms import rotated_crop_size
from bildwerk.selection import RectShape, Selection

from .conftest import noise_image


def laplacian_energy(img: Image.Image) -> float:
    arr = np.asarray(img.convert("L"), dtype=np.float64)
    lap = 4 * arr[1:-1, 1:-1] - arr[:-2, 1:-1] - arr[2:, 1:-1] - arr[1:-1, :-2] - arr[1:-1, 2:]
    return float(np.mean(lap**2))


def edge_image(blur: float = 0) -> Image.Image:
    img = Image.new("RGB", (160, 120), (40, 40, 40))
    ImageDraw.Draw(img).rectangle([50, 30, 110, 90], fill=(220, 220, 220))
    return img.filter(ImageFilter.GaussianBlur(blur)) if blur else img


SELECTION = Selection(shapes=[RectShape(x=20, y=15, w=60, h=40)])


@pytest.mark.parametrize("op_id", sorted(REGISTRY))
@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_every_operation_runs_with_defaults(op_id, mode):
    op = get_operation(op_id)
    img = noise_image(120, 80, mode)
    result = run_operation(op, img, {}, SELECTION)
    if op.kind == "new_document":
        assert isinstance(result, NewImage)
        return
    assert isinstance(result, Change)
    if op.kind == "filter" and result.kind == "patch":
        assert result.patch.size == (result.box[2] - result.box[0], result.box[3] - result.box[1])
        assert result.patch.mode == mode
    if op.kind == "filter" and op_id != "erase":
        assert result.result.mode == mode


def test_catalog_is_serializable_and_complete():
    ops = [op["id"] for cat in catalog() for op in cat["operations"]]
    assert sorted(ops) == sorted(REGISTRY)
    sharpen = next(op for cat in catalog() for op in cat["operations"] if op["id"] == "sharpen")
    assert {p["name"] for p in sharpen["params"]} == {"amount", "radius", "threshold"}


def test_param_coercion():
    p = Param("x", "X", "int", 5, min=0, max=10)
    assert p.coerce(None) == 5
    assert p.coerce("7.6") == 8
    assert p.coerce(99) == 10
    with pytest.raises(OperationError):
        p.coerce("abc")
    choice = Param("c", "C", "choice", "a", choices=(("a", "A"), ("b", "B")))
    with pytest.raises(OperationError):
        choice.coerce("z")
    color = Param("col", "Farbe", "color", "#ffffff")
    assert color.coerce("#ABCDEF") == "#abcdef"
    with pytest.raises(OperationError):
        color.coerce("red")
    assert Param("b", "B", "bool", False).coerce("true") is True


def test_sharpen_adds_detail_and_blur_removes_it():
    img = edge_image(blur=2)
    sharp = run_operation(get_operation("sharpen"), img, {"amount": 200}, None).result
    soft = run_operation(get_operation("blur"), img, {"radius": 3}, None).result
    assert laplacian_energy(sharp) > laplacian_energy(img) * 1.5
    assert laplacian_energy(soft) < laplacian_energy(img)


def test_deblur_restores_edges_without_boosting_noise():
    blurred = edge_image(blur=1.8)
    noisy = np.asarray(blurred, dtype=np.float64) + np.random.default_rng(1).normal(0, 4, (120, 160, 3))
    img = Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))
    out = run_operation(get_operation("deblur"), img, {"sigma": 1.8, "iterations": 20}, None).result

    def max_gradient(im):
        return np.abs(np.diff(np.asarray(im.convert("L"), dtype=np.float64), axis=1)).max()

    def flat_noise(im):
        return np.asarray(im.convert("L"), dtype=np.float64)[5:25, 5:40].std()

    assert max_gradient(out) > max_gradient(img) * 1.2
    assert flat_noise(out) <= flat_noise(img) * 1.1


def test_denoise_reduces_noise_in_flat_area():
    flat = np.full((80, 80, 3), 128.0) + np.random.default_rng(2).normal(0, 10, (80, 80, 3))
    img = Image.fromarray(np.clip(flat, 0, 255).astype(np.uint8))
    out = run_operation(get_operation("denoise"), img, {"strength": 60}, None).result
    assert np.asarray(out, dtype=np.float64).std() < np.asarray(img, dtype=np.float64).std() * 0.6


def test_filter_changes_only_the_selection():
    img = noise_image(120, 80)
    change = run_operation(get_operation("invert"), img, {}, SELECTION)
    assert change.kind == "patch" and change.box == (20, 15, 80, 55)
    after = img.copy()
    after.paste(change.patch, change.box[:2])
    a, b = np.asarray(img), np.asarray(after)
    outside = np.ones(a.shape[:2], bool)
    outside[15:55, 20:80] = False
    assert np.array_equal(a[outside], b[outside])
    assert np.array_equal(b[~outside], 255 - a[~outside])


def test_feathered_selection_blends_gradually():
    img = Image.new("RGB", (100, 100), (0, 0, 0))
    sel = Selection(shapes=[RectShape(x=30, y=30, w=40, h=40)], feather=8)
    change = run_operation(get_operation("fill"), img, {"color": "#ffffff"}, sel)
    after = img.copy()
    after.paste(change.patch, change.box[:2])
    row = np.asarray(after)[50, :, 0]
    assert row[50] == 255 and row[0] == 0
    assert 0 < row[30] < 255  # weicher Übergang an der Kante


def test_requires_selection():
    with pytest.raises(OperationError):
        run_operation(get_operation("crop"), noise_image(), {}, None)
    with pytest.raises(OperationError):
        run_operation(get_operation("erase"), noise_image(), {}, None)
    with pytest.raises(OperationError, match="leer"):
        run_operation(get_operation("blur"), noise_image(), {}, Selection(shapes=[RectShape(x=0, y=0, w=5, h=5, mode="subtract")]))


def test_crop_and_detail_zoom_use_selection_box():
    img = noise_image(120, 80)
    crop = run_operation(get_operation("crop"), img, {}, SELECTION)
    assert crop.kind == "replace" and crop.image.size == (60, 40)
    zoom = run_operation(get_operation("detail_zoom"), img, {"factor": "3"}, SELECTION)
    assert isinstance(zoom, NewImage) and zoom.image.size == (180, 120) and zoom.label == "Detail-Zoom"


@pytest.mark.parametrize(
    "params,size",
    [
        ({"mode": "percent", "percent": 50}, (60, 40)),
        ({"mode": "width", "width": 30}, (30, 20)),
        ({"mode": "height", "height": 160}, (240, 160)),
        ({"mode": "exact", "width": 10, "height": 70}, (10, 70)),
    ],
)
def test_resize_modes(params, size):
    assert run_operation(get_operation("resize"), noise_image(120, 80), params, None).image.size == size


def test_resize_rejects_huge_results():
    with pytest.raises(OperationError, match="zu groß"):
        run_operation(get_operation("resize"), noise_image(120, 80), {"mode": "exact", "width": 30000, "height": 30000}, None)


def test_rotate_and_flip():
    img = noise_image(120, 80)
    assert run_operation(get_operation("rotate90"), img, {"direction": "cw"}, None).image.size == (80, 120)
    flipped = run_operation(get_operation("flip"), img, {"direction": "horizontal"}, None).image
    assert np.array_equal(np.asarray(flipped), np.asarray(img)[:, ::-1])
    straight = run_operation(get_operation("rotate"), Image.new("RGB", (200, 100), "red"), {"angle": 10, "crop": True}, None).image
    assert straight.width < 200 and straight.mode == "RGB"
    corners = [straight.getpixel(p) for p in [(0, 0), (straight.width - 1, straight.height - 1)]]
    assert all(c[0] > 200 and c[1] < 60 for c in corners)  # keine weißen Ecken
    transparent = run_operation(get_operation("rotate"), img, {"angle": 30, "crop": False, "background": "transparent"}, None).image
    assert transparent.mode == "RGBA" and transparent.getpixel((0, 0))[3] == 0


def test_rotated_crop_size():
    assert rotated_crop_size(200, 100, 0) == (200, 100)
    assert rotated_crop_size(200, 100, 90) == (100, 200)
    w, h = rotated_crop_size(200, 100, 10)
    assert w < 200 and h < 100


def test_erase_makes_selection_transparent():
    change = run_operation(get_operation("erase"), noise_image(120, 80), {}, SELECTION)
    assert change.kind == "replace" and change.image.mode == "RGBA"
    alpha = np.asarray(change.image.getchannel("A"))
    assert alpha[30, 40] == 0 and alpha[0, 0] == 255


def test_pixelate_makes_uniform_blocks():
    out = run_operation(get_operation("pixelate"), noise_image(64, 64), {"block": 16}, None).image
    block = np.asarray(out)[16:32, 16:32]
    assert (block == block[0, 0]).all()


def test_adjustments_keep_alpha():
    img = noise_image(40, 40, "RGBA")
    out = run_operation(get_operation("brightness_contrast"), img, {"brightness": 50}, None).image
    assert np.array_equal(np.asarray(out.getchannel("A")), np.asarray(img.getchannel("A")))
    assert np.asarray(out.convert("RGB"), dtype=int).mean() > np.asarray(img.convert("RGB"), dtype=int).mean()


def test_fill_covers_transparent_areas():
    img = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    out = run_operation(get_operation("fill"), img, {"color": "#ff0000"}, None).image
    assert out.getpixel((5, 5)) == (255, 0, 0, 255)
