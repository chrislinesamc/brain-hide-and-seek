"""Search with SIFT keypoints: match the query box's keypoints to every section, keep only
matches that agree on one rotation + shift (RANSAC), score = how many agree (inliers).

Unlike the feature-grid search, SIFT handles any angle by itself and plain tissue gives
few keypoints, so a section can come back with no match at all. Not mirror invariant.

Fixed choices (set before any test ran, not tuned): 2x upscale like the DINOv3 features,
CLAHE contrast, Lowe ratio 0.75, RANSAC tolerance 3 thumbnail px, scale 0.8-1.25 (every
section has the same pixel size, so a big scale change is a wrong match).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from region_search.sections import Roi
from region_search.template_match import BACKGROUND_LEVEL

UPSCALE = 2
LOWE_RATIO = 0.75
RANSAC_PX = 3 * UPSCALE
MIN_SCALE, MAX_SCALE = 0.8, 1.25
MIN_MATCHES = 3  # a rotation + shift + scale needs at least 2 points; 3 to have a vote


@dataclass(frozen=True)
class SiftHit:
    inliers: int  # matches that agree on one transform; 0 = no match
    x: int  # box on the section (query box size, centred where the query centre lands)
    y: int
    angle: float  # degrees the query turned to land here


NO_HIT = SiftHit(0, -1, -1, 0.0)


def prepare(image: np.ndarray) -> np.ndarray:
    big = cv2.resize(image, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(big)


def tissue_mask(image: np.ndarray) -> np.ndarray:
    """Tissue at UPSCALE, without 1-px lines (the thumbnails' dark border, turned with the section)."""
    tissue = (image < BACKGROUND_LEVEL).astype(np.uint8)
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    tissue = cv2.dilate(tissue, np.ones((5, 5), np.uint8))
    return cv2.resize(tissue, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_NEAREST) * 255


class SiftIndex:
    def __init__(self, sections: list[int], images: list[np.ndarray]):
        self.sections = list(sections)
        self.images = list(images)
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.keypoints, self.descriptors = [], []
        for image in images:
            points, descriptors = self.sift.detectAndCompute(prepare(image), tissue_mask(image))
            self.keypoints.append(np.float32([p.pt for p in points]))
            self.descriptors.append(descriptors)

    def query_keypoints(self, section: int, roi: Roi) -> tuple[np.ndarray, np.ndarray | None]:
        """Keypoints inside the box on a stack section (thumbnail-pixel coordinates x UPSCALE)."""
        i = self.sections.index(section)
        mask = np.zeros_like(prepare(self.images[i]))
        mask[roi.y * UPSCALE : (roi.y + roi.h) * UPSCALE, roi.x * UPSCALE : (roi.x + roi.w) * UPSCALE] = 255
        mask &= tissue_mask(self.images[i])
        points, descriptors = self.sift.detectAndCompute(prepare(self.images[i]), mask)
        return np.float32([p.pt for p in points]), descriptors

    def search(self, section: int, roi: Roi, only: list[int] | None = None) -> dict[int, SiftHit]:
        """Best transform of the box onto each section (all, or `only`), as inliers + box."""
        points, descriptors = self.query_keypoints(section, roi)
        centre = np.float32([(roi.x + roi.w / 2) * UPSCALE, (roi.y + roi.h / 2) * UPSCALE, 1])
        hits = {}
        for n in only if only is not None else self.sections:
            i = self.sections.index(n)
            hits[n] = self.match(points, descriptors, self.keypoints[i], self.descriptors[i], centre, roi)
        return hits

    def match(self, points, descriptors, target_points, target_descriptors, centre, roi) -> SiftHit:
        if descriptors is None or target_descriptors is None or len(descriptors) < MIN_MATCHES or len(target_descriptors) < 2:
            return NO_HIT
        pairs = self.matcher.knnMatch(descriptors, target_descriptors, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < LOWE_RATIO * p[1].distance]
        if len(good) < MIN_MATCHES:
            return NO_HIT
        source = points[[m.queryIdx for m in good]]
        target = target_points[[m.trainIdx for m in good]]
        matrix, inliers = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX
        )
        if matrix is None:
            return NO_HIT
        scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
        count = int(inliers.sum())
        if not MIN_SCALE <= scale <= MAX_SCALE or count < MIN_MATCHES:
            return NO_HIT
        cx, cy = (matrix @ centre) / UPSCALE
        angle = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
        return SiftHit(count, round(cx - roi.w / 2), round(cy - roi.h / 2), angle)
