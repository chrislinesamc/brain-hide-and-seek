"""Measure where the thumbnails sit in the 3D volume (source of the constants in structures.py).

1. Section -> y: match every thumbnail against every coronal slice of the 400 um volume
   (both shrunk to 87 x 76, correlation), fit slice = a + b * section.
2. In-plane: phase-correlate a few thumbnails against their slice stretched to 500 x 434;
   a near-zero offset means the thumbnail spans the volume's x/z extent.

    python -I scripts/check_section_mapping.py
"""

import sys
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from region_search.sections import load_stack  # noqa: E402
from region_search.structures import SECTION_STEP_MM, SECTION_Y0_MM, THUMBNAIL_SHAPE  # noqa: E402

SMALL = (87, 76)


def unit(image: np.ndarray) -> np.ndarray:
    flat = image.astype(np.float32).ravel()
    flat -= flat.mean()
    return flat / (np.linalg.norm(flat) or 1)


def coronal(volume: np.ndarray, j: float) -> np.ndarray:
    """Slice at (fractional) y index j, rows = z with superior on top, columns = x."""
    j0 = int(j)
    t = float(j - j0)
    return ((1 - t) * volume[:, j0, :] + t * volume[:, min(j0 + 1, volume.shape[1] - 1), :]).T[::-1]


def main() -> None:
    sections, images = load_stack(ROOT / "data" / "thumbnails")
    image = nib.load(ROOT / "data" / "bigbrain3d" / "full8_400um_optbal.nii.gz")
    volume = np.asarray(image.dataobj, dtype=np.float32)
    y0, step = image.affine[1, 3], image.affine[1, 1]

    thumbs = np.stack([unit(cv2.resize(i.astype(np.float32), SMALL, interpolation=cv2.INTER_AREA)) for i in images])
    slices = np.stack([unit(cv2.resize(coronal(volume, j), SMALL, interpolation=cv2.INTER_AREA))
                       for j in range(volume.shape[1])])
    similarity = thumbs @ slices.T
    best, peak = similarity.argmax(1), similarity.max(1)
    numbers = np.array(sections)
    b, a = np.polyfit(numbers, best, 1)
    residual = np.abs(best - (a + b * numbers))
    print(f"slice = {a:.2f} + {b:.5f} * section   (median peak correlation {np.median(peak):.3f})")
    print(f"residual median {np.median(residual):.2f} slices, max {residual.max():.1f}")
    print(f"-> y = {y0 + step * a:.2f} + {step * b:.4f} * section mm   "
          f"(structures.py: {SECTION_Y0_MM} + {SECTION_STEP_MM} * section)")

    height, width = THUMBNAIL_SHAPE
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    by_section = dict(zip(sections, images))
    for n in (1001, 3001, 4501, 6501):
        stretched = cv2.resize(coronal(volume, a + b * n), (width, height), interpolation=cv2.INTER_LINEAR)
        (dx, dy), _ = cv2.phaseCorrelate(by_section[n].astype(np.float32), stretched, window)
        print(f"section {n}: in-plane offset dx {dx:+.2f} px, dy {dy:+.2f} px")


if __name__ == "__main__":
    main()
