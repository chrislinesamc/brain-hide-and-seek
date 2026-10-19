"""Baseline search: slide the ROI over each section, score with normalised cross-correlation."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from region_search.sections import Roi, crop_roi

# Sections have a pure white background (255); stained tissue, white matter included,
# sits around 160-215. Windows that are mostly background have almost no variance, so
# NCC divides noise by ~0 and can score above 1. They can never be a real match.
BACKGROUND_LEVEL = 250
MAX_BACKGROUND_FRACTION = 0.5
NO_MATCH = -1.0

UPRIGHT = (0,)
QUARTER_TURNS = (0, 90, 180, 270)

# OpenCV threads inside each call; we already run one call per section in parallel.
cv2.setNumThreads(1)


@dataclass(frozen=True)
class SectionMatch:
    section: int
    score: float  # best normalised cross-correlation on the section, -1..1
    x: int  # top-left corner of the best match
    y: int
    angle: int  # degrees counter-clockwise the query was rotated to get this match


def background_fraction(image: np.ndarray) -> float:
    return float((image >= BACKGROUND_LEVEL).mean())


def window_background_fraction(image: np.ndarray, h: int, w: int) -> np.ndarray:
    """Background fraction of every h x w window, indexed by top-left corner."""
    integral = np.pad((image >= BACKGROUND_LEVEL).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    counts = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
    return counts / (h * w)


def rotated_template(section: np.ndarray, roi: Roi, angle: int) -> np.ndarray:
    """The ROI as it would look rotated `angle` degrees counter-clockwise.

    Rotates the whole section around the ROI centre, then crops, so the corners hold
    real neighbouring tissue instead of empty padding.
    """
    if angle == 0:
        return crop_roi(section, roi)
    centre = (roi.x + roi.w / 2, roi.y + roi.h / 2)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated = cv2.warpAffine(
        section, matrix, section.shape[::-1], flags=cv2.INTER_LINEAR, borderValue=255
    )
    return crop_roi(rotated, roi)


def search_stack(
    query_section: np.ndarray,
    roi: Roi,
    sections: list[int],
    images: list[np.ndarray],
    angles: tuple[int, ...] = UPRIGHT,
) -> list[SectionMatch]:
    """Best match of the ROI on every section, over all `angles`, best first.

    One upright pass over 741 thumbnails takes ~0.35 s on an M1 Pro (8 threads).
    """
    templates = [(angle, rotated_template(query_section, roi, angle)) for angle in angles]
    if background_fraction(templates[0][1]) > MAX_BACKGROUND_FRACTION:
        raise ValueError("ROI is mostly background, draw it over tissue")

    def best_match(number: int, image: np.ndarray) -> SectionMatch:
        excluded = window_background_fraction(image, roi.h, roi.w) > MAX_BACKGROUND_FRACTION
        best = SectionMatch(number, NO_MATCH, 0, 0, 0)
        for angle, template in templates:
            response = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            response[excluded | ~np.isfinite(response)] = NO_MATCH
            y, x = np.unravel_index(np.argmax(response), response.shape)
            score = float(np.clip(response[y, x], -1.0, 1.0))
            if score > best.score:
                best = SectionMatch(number, score, int(x), int(y), angle)
        return best

    with ThreadPoolExecutor(max_workers=8) as pool:
        matches = list(pool.map(best_match, sections, images))
    return sorted(matches, key=lambda m: m.score, reverse=True)
