import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from region_search.feature_search import FeatureIndex, turned_roi  # noqa: E402
from region_search.sections import Roi  # noqa: E402
from region_search.template_match import QUARTER_TURNS  # noqa: E402

ROWS, COLS, DIM = 10, 12, 16


def unit_noise(rng, *shape):
    grid = rng.normal(size=shape).astype(np.float32)
    return (grid / np.linalg.norm(grid, axis=-1, keepdims=True)).astype(np.float16)


def tissue(n):
    return [np.full((ROWS * 8, COLS * 8), 180, dtype=np.uint8) for _ in range(n)]


def test_finds_the_planted_region_and_maps_back_to_pixels():
    rng = np.random.default_rng(0)
    features = unit_noise(rng, 3, ROWS, COLS, DIM)
    features[2, 5:8, 6:9] = features[0, 1:4, 2:5]  # section 30 holds section 10's patches 3 cols / 4 rows away
    index = FeatureIndex(features, [10, 20, 30], tissue(3))

    ranked = index.search(10, Roi(x=2 * 8 + 3, y=1 * 8 + 5, w=20, h=19))  # spans patches (2..4, 1..3)

    assert [m.section for m in ranked][:2] == [10, 30]
    assert ranked[0].score == pytest.approx(1.0, abs=1e-2)
    assert ranked[1].score == pytest.approx(1.0, abs=1e-2)
    assert (ranked[1].x, ranked[1].y) == (6 * 8 + 3, 5 * 8 + 5)  # same offset inside the patch
    assert ranked[2].score < 0.5


def test_several_boxes_rank_by_their_mean_and_each_finds_its_own_spot():
    rng = np.random.default_rng(3)
    features = unit_noise(rng, 3, ROWS, COLS, DIM)
    left, right = features[0, 1:4, 1:4], features[0, 5:8, 7:10]
    features[1, 1:4, 2:5] = left  # section 20 holds both parts, shifted one patch right
    features[1, 5:8, 8:11] = right
    features[2, 1:4, 2:5] = left  # section 30 only the left part
    index = FeatureIndex(features, [10, 20, 30], tissue(3))

    ranked = index.search_boxes(10, [Roi(8, 8, 24, 24), Roi(56, 40, 24, 24)])

    assert [m.section for m in ranked] == [10, 20, 30]
    both = ranked[1]
    assert both.score == pytest.approx(1.0, abs=1e-2)
    assert [(hits[0].x, hits[0].y) for hits in both.boxes] == [(16, 8), (64, 40)]
    assert 0.4 < ranked[2].score < 0.8  # one part found, the other not


def test_background_windows_never_match():
    rng = np.random.default_rng(1)
    features = unit_noise(rng, 2, ROWS, COLS, DIM)
    features[1, 0:3, 0:3] = features[0, 4:7, 4:7]
    images = tissue(2)
    images[1][:40, :40] = 255  # the copy sits on blank background
    index = FeatureIndex(features, [1, 2], images)

    ranked = index.search(1, Roi(x=32, y=32, w=24, h=24))
    match = next(m for m in ranked if m.section == 2)
    assert match.score < 0.5


def test_rejects_mostly_background_roi():
    rng = np.random.default_rng(2)
    images = tissue(1)
    images[0][:, :] = 255
    index = FeatureIndex(unit_noise(rng, 1, ROWS, COLS, DIM), [1], images)
    with pytest.raises(ValueError, match="background"):
        index.search(1, Roi(x=0, y=0, w=16, h=16))


def block_extractor(dim=32, seed=3):
    """Stand-in for DINOv3: each 8x8 pixel block -> fixed random projection, L2-normalised."""
    projection = np.random.default_rng(seed).normal(size=(64, dim)).astype(np.float32)

    def extract(image):
        rows, cols = image.shape[0] // 8, image.shape[1] // 8
        blocks = image[: rows * 8, : cols * 8].reshape(rows, 8, cols, 8).transpose(0, 2, 1, 3)
        grid = (blocks.reshape(rows, cols, 64).astype(np.float32) - 188) @ projection  # 188 = texture mean
        return (grid / np.linalg.norm(grid, axis=-1, keepdims=True)).astype(np.float16)

    return extract


def test_quarter_turn_finds_a_rotated_copy():
    rng = np.random.default_rng(4)
    upright = rng.integers(160, 216, size=(80, 96), dtype=np.uint8)  # textured "tissue"
    turned = np.ascontiguousarray(np.rot90(upright, 1))  # 96 x 80
    canvas = rng.integers(160, 216, size=(96, 96), dtype=np.uint8)  # unrelated texture around it
    canvas[:, :80] = turned
    images = [np.vstack([upright, rng.integers(160, 216, size=(16, 96), dtype=np.uint8)]), canvas]
    extract = block_extractor()
    index = FeatureIndex(np.stack([extract(im) for im in images]), [1, 2], images, extract)
    roi = Roi(x=40, y=24, w=16, h=16)

    upright_only = {m.section: m for m in index.search(1, roi)}
    rotated = {m.section: m for m in index.search(1, roi, QUARTER_TURNS)}

    assert upright_only[2].score < 0.5
    assert rotated[2].score == pytest.approx(1.0, abs=1e-2)
    assert rotated[2].angle == 90
    expected = turned_roi(roi, images[0].shape, 1)
    assert (rotated[2].x, rotated[2].y) == (expected.x, expected.y)


def test_turned_roi_keeps_size_and_centre():
    roi = Roi(x=10, y=20, w=30, h=10)  # centre (25, 25) in a 100 wide x 60 tall image
    assert turned_roi(roi, (60, 100), 0) == roi
    assert turned_roi(roi, (60, 100), 1) == Roi(10, 70, 30, 10)  # centre -> (25, 75)
    assert turned_roi(roi, (60, 100), 2) == Roi(60, 30, 30, 10)  # centre -> (75, 35)
    assert turned_roi(roi, (60, 100), 4) == roi


def test_rotation_without_extractor_is_refused():
    rng = np.random.default_rng(5)
    index = FeatureIndex(unit_noise(rng, 1, ROWS, COLS, DIM), [1], tissue(1))
    with pytest.raises(ValueError, match="extractor"):
        index.search(1, Roi(x=8, y=8, w=16, h=16), QUARTER_TURNS)


def test_mirror_finds_the_twin_on_the_same_section():
    rng = np.random.default_rng(6)
    half = rng.integers(160, 216, size=(80, 48), dtype=np.uint8)
    twins = np.hstack([half, np.fliplr(half)])  # left part and its mirror image, like two hippocampi
    other = rng.integers(160, 216, size=(80, 96), dtype=np.uint8)
    images = [twins, other]
    extract = block_extractor()
    index = FeatureIndex(np.stack([extract(im) for im in images]), [1, 2], images, extract)
    roi = Roi(x=8, y=24, w=16, h=16)

    plain = {s.section: s for s in index.search_hits(1, roi, max_hits=2, extra_margin=0.05)}
    mirrored = {s.section: s for s in index.search_hits(1, roi, mirror=True, max_hits=2, extra_margin=0.05)}

    assert len(plain[1].hits) == 1  # without mirroring the twin does not look alike
    first, twin = mirrored[1].hits
    assert (first.x, first.y, first.mirrored) == (8, 24, False)
    assert (twin.x, twin.y, twin.mirrored) == (96 - 8 - 16, 24, True)
    assert twin.score == pytest.approx(1.0, abs=1e-2)



def test_window_scores_equal_conv2d():
    import torch
    import torch.nn.functional as F

    from region_search.feature_search import window_scores

    generator = torch.Generator().manual_seed(0)
    features = torch.randn(5, 16, 12, 15, generator=generator)
    weights = torch.randn(1, 16, 4, 6, generator=generator)
    expected = F.conv2d(features, weights)[:, 0]
    assert torch.allclose(window_scores(features, weights), expected, atol=1e-4)
