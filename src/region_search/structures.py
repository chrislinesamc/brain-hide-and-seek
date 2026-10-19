"""Where named brain structures sit in the BigBrain 2015 section stack.

Ground truth for evaluation: a structure's labelled points (BigBrain histological space,
mm) give the range of sections it spans and a query box on one section.

Sources (BigBrain FTP, scripts/download_structures.py):
- hippocampus: DeKraker et al. 2020 mid-surfaces, segmented on BigBrain itself.
- cortical areas: DKT atlas labels mapped onto the BigBrain white surface (Lewis et al.,
  MSM registration). Automatic, not hand-marked; the white surface is the grey/white
  boundary, so the cortex lies just outside these points.
"""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from region_search.sections import Roi

# Section -> y. Measured by scripts/check_section_mapping.py: every 10th thumbnail matched
# against the coronal slices of full8_400um_optbal.nii.gz fits slice = 7.72 + 0.04998 *
# section (median residual 0.25 slice = ~5 sections, max 0.8), no flips.
SECTION_Y0_MM = -69.92
SECTION_STEP_MM = 0.02

# Thumbnail pixel -> x/z: the 500 x 434 thumbnail spans the volume's x and z extent
# (column = +x, row 0 = top = +z). Same script: offset <= 0.74 px on 7 sections.
THUMBNAIL_SHAPE = (434, 500)
X_LEFT_MM, MM_PER_PX_X = -70.867, 139.2 / 500
Z_TOP_MM, MM_PER_PX_Z = 62.222, 121.2 / 434

SLAB_MM = 0.5  # points within this distance of a section's plane count as on it

# display name -> DKT label, or None for the hippocampus surfaces
STRUCTURES = {
    "hippocampus": None,
    "insula": "insula",
    "Heschl's gyrus": "transversetemporal",
    "calcarine cortex": "pericalcarine",
}

# More DKT areas, only for scripts/fixed_test.py: never used to tune anything, so a result
# on them is a fair test (lingual + supramarginal = the one set thresholds are picked on).
FIXED_TEST_REGIONS = {
    "superior temporal gyrus": "superiortemporal",
    "cuneus": "cuneus",
    "precentral gyrus": "precentral",
    "lingual gyrus": "lingual",
    "supramarginal gyrus": "supramarginal",
}


@dataclass(frozen=True)
class Structure:
    name: str
    points: np.ndarray  # N x 3, x/y/z in mm
    left: np.ndarray  # N bools, True = left hemisphere (image left on the thumbnails)


def load_structure(name: str, folder: Path) -> Structure:
    known = {**STRUCTURES, **FIXED_TEST_REGIONS}
    if name not in known:
        raise ValueError(f"unknown structure {name!r}, known: {', '.join(known)}")
    label = known[name]
    parts = []
    for hemi in "LR":
        if label is None:
            points = surface_points(folder / f"sub-bbhist_hemi-{hemi}_midSurf.surf.gii")
        else:
            h = hemi.lower()
            points = surface_points(folder / f"{h}h.white.anat.surf.gii")
            labels = nib.load(folder / f"{h}h.DKTatlas40.label.gii")
            ids = {text: key for key, text in labels.labeltable.get_labels_as_dict().items()}
            points = points[labels.darrays[0].data == ids[label]]
        parts.append((points, np.full(len(points), hemi == "L")))
    return Structure(name, np.concatenate([p for p, _ in parts]), np.concatenate([s for _, s in parts]))


def surface_points(path: Path) -> np.ndarray:
    return np.asarray(nib.load(path).darrays[0].data, dtype=np.float64)


def section_at(y_mm: float) -> float:
    return (y_mm - SECTION_Y0_MM) / SECTION_STEP_MM


def section_range(structure: Structure) -> tuple[int, int]:
    """First and last section the structure reaches (either hemisphere)."""
    y = structure.points[:, 1]
    return round(section_at(y.min())), round(section_at(y.max()))


def query_section(structure: Structure, sections: list[int]) -> int:
    """The stack section nearest the middle of the structure's range: where a by-name search starts."""
    first, last = section_range(structure)
    return min(sections, key=lambda n: abs(n - (first + last) / 2))


# Optional `to_stack` below: 2 x 3 matrix from thumbnail pixel to the pixel of a transformed
# stack (region_search.scramble); `shape` is then that stack's image shape.


def query_box(
    structure: Structure, section: int, left: bool = True, pad: int = 8,
    to_stack: np.ndarray | None = None, shape: tuple[int, int] = THUMBNAIL_SHAPE,
) -> Roi:
    """Box around the structure's points on one section, padded by `pad` px."""
    selected = on_section(structure, section) & (structure.left == left)
    if not selected.any():
        side = "left" if left else "right"
        raise ValueError(f"{structure.name} ({side}) has no points on section {section}")
    cols, rows = thumbnail_pixels(structure.points[selected], to_stack)
    height, width = shape
    x0, y0 = max(cols.min() - pad, 0), max(rows.min() - pad, 0)
    x1, y1 = min(cols.max() + 1 + pad, width), min(rows.max() + 1 + pad, height)
    return Roi(int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def box_hits(structure: Structure, section: int, roi: Roi, to_stack: np.ndarray | None = None) -> bool | None:
    """Whether the box covers any of the structure's points on the section (either hemisphere).

    None when the structure has no points on that section, so there is nothing to hit.
    """
    selected = on_section(structure, section)
    if not selected.any():
        return None
    cols, rows = thumbnail_pixels(structure.points[selected], to_stack)
    inside = (cols >= roi.x) & (cols < roi.x + roi.w) & (rows >= roi.y) & (rows < roi.y + roi.h)
    return bool(inside.any())


def on_section(structure: Structure, section: int) -> np.ndarray:
    y = SECTION_Y0_MM + SECTION_STEP_MM * section
    return np.abs(structure.points[:, 1] - y) <= SLAB_MM


def thumbnail_pixels(points: np.ndarray, to_stack: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Column and row of the pixel each point falls in (thumbnail, or stack via `to_stack`)."""
    # pixel i covers [i - 0.5, i + 0.5) in these coordinates
    cols = (points[:, 0] - X_LEFT_MM) / MM_PER_PX_X - 0.5
    rows = (Z_TOP_MM - points[:, 2]) / MM_PER_PX_Z - 0.5
    if to_stack is not None:
        cols, rows = to_stack @ np.stack([cols, rows, np.ones_like(cols)])
    return np.round(cols).astype(int), np.round(rows).astype(int)
