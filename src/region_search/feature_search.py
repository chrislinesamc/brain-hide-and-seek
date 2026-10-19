"""Search in feature space: slide the ROI's grid of patch features over every section.

Score at a position = mean cosine similarity between the ROI's patch features and the
patch features under it, so -1..1 like the NCC baseline.

Rotation and mirroring: patch features are not rotation or mirror invariant, so turning
the feature grid would not mean turning the tissue. Instead the query section itself is
flipped and/or turned by exact quarter turns and its features re-extracted (one model pass
per orientation); the stack stays as it is.

Several hits per section: after the best spot, the next best spots that do not overlap it
(e.g. the left hippocampus box also finding the right one, with mirroring on).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from region_search.features import PATCH, UPSCALE, pick_device
from region_search.sections import Roi
from region_search.template_match import (
    BACKGROUND_LEVEL,
    MAX_BACKGROUND_FRACTION,
    NO_MATCH,
    UPRIGHT,
    SectionMatch,
)

PIXELS_PER_PATCH = PATCH // UPSCALE  # one patch covers 8x8 thumbnail pixels
CHUNK = 64  # sections per scoring call; keeps the float32 copy of one chunk small

# section image -> rows x cols x dim grid, laid out like the precomputed stack
Extractor = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Hit:
    x: int  # top-left corner of the box on the section
    y: int
    score: float
    angle: int  # degrees counter-clockwise the query was turned to get this hit
    mirrored: bool  # query flipped left-right (before turning)


@dataclass(frozen=True)
class SectionHits:
    section: int
    hits: list[Hit]  # best first, boxes do not overlap

    @property
    def score(self) -> float:
        return self.hits[0].score


@dataclass(frozen=True)
class MultiBoxMatch:
    section: int
    score: float  # mean of the boxes' best scores on this section
    boxes: list[list[Hit]]  # hits of each query box, in query order


class FeatureIndex:
    def __init__(
        self,
        features: np.ndarray,
        sections: list[int],
        images: list[np.ndarray],
        extract: Extractor | None = None,
    ):
        if features.shape[0] != len(sections) or len(sections) != len(images):
            raise ValueError("features, sections and images must cover the same sections")
        self.sections = list(sections)
        self.images = list(images)
        self.extract = extract  # only needed for turned or mirrored queries
        self.device = pick_device()
        # N x dim x rows x cols, the layout conv2d wants
        self.features = torch.from_numpy(features).permute(0, 3, 1, 2).contiguous().to(self.device)
        rows, cols = features.shape[1:3]
        self.background = torch.from_numpy(
            np.stack([patch_background(image, rows, cols) for image in images])
        ).to(self.device)

    @classmethod
    def load(
        cls, path: Path, images_by_section: dict[int, np.ndarray], extract: Extractor | None = None
    ) -> "FeatureIndex":
        """Load a stack saved by scripts/precompute_features.py."""
        sections = [int(n) for n in np.load(path.with_name(path.stem + "-sections.npy"))]
        missing = [n for n in sections if n not in images_by_section]
        if missing:
            raise ValueError(f"{len(missing)} feature sections have no thumbnail, e.g. {missing[0]}")
        return cls(np.load(path), sections, [images_by_section[n] for n in sections], extract)

    def search(self, query_section: int, roi: Roi, angles: tuple[int, ...] = UPRIGHT) -> list[SectionMatch]:
        """Best match of the ROI on every section, over all `angles`, best first."""
        return [
            SectionMatch(s.section, s.score, s.hits[0].x, s.hits[0].y, s.hits[0].angle)
            for s in self.search_hits(query_section, roi, angles)
        ]

    @torch.inference_mode()
    def search_hits(
        self, query_section: int, roi: Roi, angles: tuple[int, ...] = UPRIGHT, mirror: bool = False,
        max_hits: int = 1, extra_margin: float = 0.0, extra_floor: float = NO_MATCH,
    ) -> list[SectionHits]:
        """Up to `max_hits` non-overlapping hits of the ROI on every section, over all
        `angles` (and their mirror images with `mirror`), sections best first.

        Hits after the first are kept only within `extra_margin` of the section's best score
        and at or above `extra_floor`.
        """
        if any(angle % 90 for angle in angles):
            raise ValueError(f"only quarter turns are supported, got {angles}")
        orientations = [(a, m) for m in ((False, True) if mirror else (False,)) for a in angles]
        if any(o != (0, False) for o in orientations) and self.extract is None:
            raise ValueError("turned or mirrored search needs a feature extractor")
        query = self.sections.index(query_section)
        kernels = [(o, self.kernel(query, roi, *o)) for o in orientations]
        if kernels[0][1] is None:
            raise ValueError("ROI is mostly background, draw it over tissue")

        n = len(self.sections)
        candidates = [[] for _ in range(n)]  # per section: (score, x, y, angle, mirrored)
        for (angle, mirrored), kernel in kernels:
            if kernel is None:  # this orientation puts the box mostly on background
                continue
            weights, dx, dy = kernel
            h, w = weights.shape[-2:]
            for start in range(0, n, CHUNK):
                response = window_scores(self.features[start : start + CHUNK].float(), weights)
                window_background = F.avg_pool2d(self.background[start : start + CHUNK, None], (h, w), stride=1)[:, 0]
                response[window_background > MAX_BACKGROUND_FRACTION] = NO_MATCH
                for i, peaks in enumerate(top_peaks(response, max_hits, h, w)):
                    candidates[start + i] += [
                        (score, col * PIXELS_PER_PATCH + dx, row * PIXELS_PER_PATCH + dy, angle, mirrored)
                        for score, row, col in peaks
                    ]

        results = [
            SectionHits(number, keep_apart(found, roi, max_hits, extra_margin, extra_floor))
            for number, found in zip(self.sections, candidates)
        ]
        return sorted(results, key=lambda s: s.score, reverse=True)

    def search_boxes(
        self, query_section: int, rois: list[Roi], angles: tuple[int, ...] = UPRIGHT, mirror: bool = False,
        max_hits: int = 1, extra_margin: float = 0.0, extra_floor: float = NO_MATCH,
    ) -> list[MultiBoxMatch]:
        """Several boxes from one section searched as one query (e.g. a structure's left and
        right part): each box finds its own hits, a section ranks by the boxes' mean best score."""
        if not rois:
            raise ValueError("need at least one box")
        per_box = [
            {
                s.section: s.hits
                for s in self.search_hits(query_section, roi, angles, mirror, max_hits, extra_margin, extra_floor)
            }
            for roi in rois
        ]
        matches = [
            MultiBoxMatch(n, float(np.mean([box[n][0].score for box in per_box])), [box[n] for box in per_box])
            for n in self.sections
        ]
        return sorted(matches, key=lambda m: m.score, reverse=True)

    def kernel(self, query: int, roi: Roi, angle: int, mirrored: bool = False) -> tuple[torch.Tensor, int, int] | None:
        """The ROI's patch features as a conv kernel, from the flipped and/or turned query section.

        Returns (weights, dx, dy), dx/dy = the box's offset inside its first patch, or None
        when the box covers mostly background.
        """
        if angle == 0 and not mirrored:
            grid, background, box = self.features[query], self.background[query], roi
        else:
            image, box = self.images[query], roi
            if mirrored:
                image = np.fliplr(image)
                box = Roi(image.shape[1] - roi.x - roi.w, roi.y, roi.w, roi.h)
            turns = angle // 90
            box = turned_roi(box, image.shape, turns)
            image = np.ascontiguousarray(np.rot90(image, turns))
            features = self.extract(image)
            grid = torch.from_numpy(features).permute(2, 0, 1).to(self.device)
            background = torch.from_numpy(patch_background(image, *features.shape[:2])).to(self.device)
        col0, row0, w, h = patch_box(box, *grid.shape[-2:])
        if background[row0 : row0 + h, col0 : col0 + w].mean() > MAX_BACKGROUND_FRACTION:
            return None
        weights = grid[None, :, row0 : row0 + h, col0 : col0 + w].float() / (w * h)
        return weights, box.x - col0 * PIXELS_PER_PATCH, box.y - row0 * PIXELS_PER_PATCH


def window_scores(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """F.conv2d(features, weights)[:, 0], without its unrolled copy of every window.

    CPU conv2d unrolled ~2 GB per 4 sections for the largest boxes (the fixed test was
    killed or swapped for an hour); on MPS one big conv tripped the GPU watchdog and gave
    corrupt scores. Here: one matmul per chunk (each section's features against every
    kernel cell), then the shifted cell maps are added up. B x rows x cols out.
    """
    _, dim, rows, cols = features.shape
    h, w = weights.shape[-2:]
    cells = weights[0].reshape(dim, h * w).T  # (h*w) x dim
    per_cell = (cells @ features.flatten(2)).view(-1, h, w, rows, cols)
    out = torch.zeros(features.shape[0], rows - h + 1, cols - w + 1, device=features.device)
    for i in range(h):
        for j in range(w):
            out += per_cell[:, i, j, i : i + rows - h + 1, j : j + cols - w + 1]
    return out


def top_peaks(response: torch.Tensor, k: int, h: int, w: int) -> list[list[tuple[float, int, int]]]:
    """Per map in the B x rows x cols response: the k best (score, row, col), each window
    (h x w cells) not overlapping an earlier one. Exhausted maps stop early."""
    response = response.clone()
    rows = torch.arange(response.shape[1], device=response.device)[None, :, None]
    cols = torch.arange(response.shape[2], device=response.device)[None, None, :]
    found = []
    for _ in range(k):
        score, flat = response.flatten(1).max(dim=1)
        row, col = flat // response.shape[2], flat % response.shape[2]
        found.append((score.cpu().numpy(), row.cpu().numpy(), col.cpu().numpy()))
        overlaps = ((rows - row[:, None, None]).abs() < h) & ((cols - col[:, None, None]).abs() < w)
        response[overlaps] = NO_MATCH
    peaks = []
    for b in range(response.shape[0]):
        # the first peak always counts (NO_MATCH if all background), later ones only if real
        peaks.append([(float(s[b]), int(r[b]), int(c[b])) for j, (s, r, c) in enumerate(found) if j == 0 or s[b] > NO_MATCH])
    return peaks


def keep_apart(candidates, roi: Roi, max_hits: int, extra_margin: float, extra_floor: float) -> list[Hit]:
    """Best candidates whose boxes don't overlap, across orientations; extras near the best."""
    kept: list[Hit] = []
    for score, x, y, angle, mirrored in sorted(candidates, key=lambda c: c[0], reverse=True):
        if len(kept) == max_hits:
            break
        if kept and (score < kept[0].score - extra_margin or score < extra_floor):
            break
        if any(abs(x - k.x) < roi.w and abs(y - k.y) < roi.h for k in kept):
            continue
        kept.append(Hit(int(x), int(y), float(np.clip(score, -1.0, 1.0)), int(angle), bool(mirrored)))
    return kept


def patch_box(roi: Roi, rows: int, cols: int) -> tuple[int, int, int, int]:
    """ROI in patch coordinates: top-left patch (col, row) and size, covering the whole ROI."""
    col0, row0 = roi.x // PIXELS_PER_PATCH, roi.y // PIXELS_PER_PATCH
    col1 = math.ceil((roi.x + roi.w) / PIXELS_PER_PATCH)
    row1 = math.ceil((roi.y + roi.h) / PIXELS_PER_PATCH)
    if col1 > cols or row1 > rows:
        raise ValueError(f"ROI {roi} falls outside the {cols}x{rows} patch grid")
    return col0, row0, col1 - col0, row1 - row0


def turned_roi(roi: Roi, shape: tuple[int, int], turns: int) -> Roi:
    """Where the ROI's centre lands after np.rot90(image, turns), same w x h around it.

    Keeping the box size (not swapping w/h) matches the pixel search: the box shows the
    turned tissue and match boxes in the viewer keep the drawn size. Clamped to the image.
    """
    height, width = shape
    cx, cy = roi.x + roi.w / 2, roi.y + roi.h / 2
    for _ in range(turns % 4):  # one counter-clockwise quarter turn: (x, y) -> (y, W - x)
        cx, cy = cy, width - cx
        height, width = width, height
    x = min(max(round(cx - roi.w / 2), 0), width - roi.w)
    y = min(max(round(cy - roi.h / 2), 0), height - roi.h)
    return Roi(x, y, roi.w, roi.h)


def patch_background(image: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Fraction of background pixels under each patch, rows x cols float32."""
    padded = np.ones((rows * PIXELS_PER_PATCH, cols * PIXELS_PER_PATCH), dtype=np.float32)
    height, width = image.shape
    padded[:height, :width] = image >= BACKGROUND_LEVEL
    return padded.reshape(rows, PIXELS_PER_PATCH, cols, PIXELS_PER_PATCH).mean(axis=(1, 3))
