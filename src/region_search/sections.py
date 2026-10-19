"""Load a stack of coronal section images and crop ROIs from them."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage.io import imread

SECTION_NUMBER = re.compile(r"pm(\d{4})o\.png$")


@dataclass(frozen=True)
class Roi:
    """Axis-aligned box in pixel coordinates of one section."""

    x: int
    y: int
    w: int
    h: int


def load_stack(folder: Path) -> tuple[list[int], list[np.ndarray]]:
    """Return section numbers and 2D grayscale images, sorted by section number."""
    paths = sorted(folder.glob("pm*o.png"))
    if not paths:
        raise FileNotFoundError(f"no pm####o.png sections in {folder}")
    numbers = [int(SECTION_NUMBER.search(p.name).group(1)) for p in paths]
    images = [imread(p) for p in paths]
    return numbers, images


def crop_roi(section: np.ndarray, roi: Roi) -> np.ndarray:
    height, width = section.shape
    if roi.w <= 0 or roi.h <= 0 or roi.x < 0 or roi.y < 0:
        raise ValueError(f"invalid ROI {roi}")
    if roi.x + roi.w > width or roi.y + roi.h > height:
        raise ValueError(f"ROI {roi} falls outside a {width}x{height} section")
    return section[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
