import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from region_search.sections import Roi  # noqa: E402
from region_search.template_match import (  # noqa: E402
    QUARTER_TURNS,
    search_stack,
    window_background_fraction,
)


def tissue(rng, shape):
    """Stain-like texture: blurred noise in the tissue grey range (160-215)."""
    noise = cv2.GaussianBlur(rng.random(shape).astype(np.float32), (0, 0), 2)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return (160 + 55 * noise).astype(np.uint8)


def test_finds_the_roi_where_it_was_cut():
    rng = np.random.default_rng(0)
    image = np.full((100, 120), 255, np.uint8)
    image[20:80, 20:100] = tissue(rng, (60, 80))
    [match] = search_stack(image, Roi(40, 30, 30, 20), [1], [image])
    assert (match.x, match.y, match.angle) == (40, 30, 0)
    assert match.score == pytest.approx(1.0, abs=1e-4)


def test_near_flat_background_never_wins():
    # Regression: background with faint grain used to score above 1 (seen: 1.05).
    rng = np.random.default_rng(1)
    query = tissue(rng, (40, 40))
    grainy_background = np.full((60, 60), 255, np.uint8)
    grainy_background[rng.random((60, 60)) < 0.02] = 254
    unrelated_tissue = tissue(rng, (60, 60))

    ranked = search_stack(query, Roi(10, 10, 20, 20), [1, 2], [grainy_background, unrelated_tissue])

    assert all(-1.0 <= m.score <= 1.0 for m in ranked)
    assert ranked[0].section == 2


def test_rejects_a_box_drawn_on_background():
    blank = np.full((60, 60), 255, np.uint8)
    with pytest.raises(ValueError, match="background"):
        search_stack(blank, Roi(10, 10, 20, 20), [1], [blank])


def test_rotation_finds_a_quarter_turned_section():
    rng = np.random.default_rng(3)
    query = tissue(rng, (160, 160))
    roi = Roi(60, 65, 40, 30)
    centre = (roi.x + roi.w / 2, roi.y + roi.h / 2)
    turned = cv2.warpAffine(query, cv2.getRotationMatrix2D(centre, 90, 1.0), (160, 160), borderValue=255)

    [upright] = search_stack(query, roi, [1], [turned])
    [rotated] = search_stack(query, roi, [1], [turned], QUARTER_TURNS)

    assert rotated.angle == 90
    assert (rotated.x, rotated.y) == (roi.x, roi.y)
    assert rotated.score > 0.95 > upright.score


def test_window_background_fraction_matches_brute_force():
    rng = np.random.default_rng(2)
    image = np.where(rng.random((15, 18)) < 0.4, 255, 180).astype(np.uint8)
    fractions = window_background_fraction(image, 4, 5)
    assert fractions.shape == (12, 14)
    for y, x in [(0, 0), (5, 7), (11, 13)]:
        assert fractions[y, x] == pytest.approx((image[y : y + 4, x : x + 5] >= 250).mean())
