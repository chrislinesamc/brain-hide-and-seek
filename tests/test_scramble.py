import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from region_search.scramble import CANVAS, scramble_section  # noqa: E402


def test_the_matrix_says_where_each_pixel_went():
    image = np.full((434, 500), 255, np.uint8)
    image[100:300, 150:400] = 190  # tissue
    image[120:124, 170:174] = 0  # a dark mark inside it
    for seed in range(5):
        canvas, matrix = scramble_section(image, np.random.default_rng(seed))
        assert canvas.shape == (CANVAS, CANVAS)
        expected = matrix @ np.array([171.5, 121.5, 1.0])  # centre of the mark
        rows, cols = np.where(canvas < 60)
        assert np.hypot(cols.mean() - expected[0], rows.mean() - expected[1]) < 1.5


def test_tissue_stays_on_the_canvas():
    image = np.full((434, 500), 255, np.uint8)
    image[0:433, 8:491] = 190  # the largest tissue extent in the stack, 483 x 433
    for seed in range(20):
        canvas, _ = scramble_section(image, np.random.default_rng(seed))
        assert (canvas < 250).sum() >= 0.99 * 483 * 433
