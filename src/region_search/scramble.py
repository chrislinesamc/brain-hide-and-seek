"""Undo the stack's alignment: turn, tilt and shift every section at random.

BigBrain sections are registered, so a structure sits in nearly the same spot from one
section to the next and "same box, nearby sections" already finds it. Real stacks of
individual slices are not aligned; a scrambled stack tests search under that condition.
"""

import cv2
import numpy as np

from region_search.template_match import BACKGROUND_LEVEL

CANVAS = 640  # fits the largest tissue (483 x 433 px) turned, tilted and shifted
MAX_TILT = 10.0  # degrees, on top of a random quarter turn
MAX_SHIFT = 40  # px, from the canvas centre
BACKGROUND = 255


def scramble_section(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Square canvas with the section turned, tilted and shifted, plus the 2 x 3 matrix
    that maps a section pixel (x, y) to its canvas pixel."""
    rows, cols = np.where(image < BACKGROUND_LEVEL)
    if rows.size == 0:
        raise ValueError("section has no tissue")
    centre = ((cols.min() + cols.max()) / 2, (rows.min() + rows.max()) / 2)
    angle = 90 * rng.integers(4) + rng.uniform(-MAX_TILT, MAX_TILT)
    shift = rng.uniform(-MAX_SHIFT, MAX_SHIFT, size=2)
    # turn about the tissue centre, then move that centre to canvas centre + shift
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    matrix[:, 2] += np.array([CANVAS / 2, CANVAS / 2]) + shift - np.array(centre)
    canvas = cv2.warpAffine(
        image, matrix, (CANVAS, CANVAS), flags=cv2.INTER_LINEAR, borderValue=BACKGROUND
    )
    return canvas, matrix
