import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from region_search.sections import Roi  # noqa: E402
from region_search.sift_search import SiftIndex  # noqa: E402


def textured_section(seed: int) -> np.ndarray:
    """Tissue (160-215) with blob texture SIFT can grip, on white background."""
    noise = np.random.default_rng(seed).normal(size=(434, 500)).astype(np.float32)
    texture = cv2.GaussianBlur(noise, (0, 0), 3)
    image = np.full((434, 500), 255, np.uint8)
    tissue = np.clip(188 + 400 * texture, 160, 215).astype(np.uint8)
    image[60:380, 60:440] = tissue[60:380, 60:440]
    return image


def test_finds_the_box_on_a_turned_copy_at_any_angle():
    image = textured_section(0)
    matrix = cv2.getRotationMatrix2D((250, 220), 37, 1.0)
    turned = cv2.warpAffine(image, matrix, (500, 434), flags=cv2.INTER_LINEAR, borderValue=255)
    plain = np.full((434, 500), 255, np.uint8)
    plain[60:380, 60:440] = 190
    index = SiftIndex([1, 2, 3], [image, turned, plain])
    roi = Roi(200, 170, 50, 50)

    hits = index.search(1, roi)

    centre = matrix @ np.array([225.0, 195.0, 1.0])
    hit = hits[2]
    assert hit.inliers >= 10
    assert abs(hit.x + roi.w / 2 - centre[0]) <= 3 and abs(hit.y + roi.h / 2 - centre[1]) <= 3
    assert abs(hit.angle - (-37)) <= 3  # image y points down, so the turn reads negative
    assert hits[3].inliers == 0  # plain tissue: no match, not a weak one


def test_unrelated_texture_is_no_match():
    index = SiftIndex([1, 2], [textured_section(0), textured_section(1)])
    assert index.search(1, Roi(200, 170, 50, 50))[2].inliers < 5
