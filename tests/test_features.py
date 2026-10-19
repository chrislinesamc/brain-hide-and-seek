import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from region_search.features import PATCH, UPSCALE, prepare  # noqa: E402

NO_NORMALISATION = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


def test_thumbnail_is_upscaled_and_padded_to_whole_patches():
    image = np.full((434, 500), 180, dtype=np.uint8)
    tensor = prepare(image, *NO_NORMALISATION)
    assert tensor.shape == (1, 3, 880, 1008)  # 868 -> 880, 1000 -> 1008
    assert tensor.shape[-2] % PATCH == 0 and tensor.shape[-1] % PATCH == 0


def test_padding_is_background_and_pixels_keep_their_place():
    image = np.full((434, 500), 180, dtype=np.uint8)
    image[100, 200] = 0
    tensor = prepare(image, *NO_NORMALISATION)[0, 0]
    assert tensor[-1, -1].item() == pytest.approx(1.0)  # padded corner = white background
    y, x = 100 * UPSCALE, 200 * UPSCALE
    assert tensor[y : y + UPSCALE, x : x + UPSCALE].min().item() < 180 / 255


def test_rejects_colour_or_float_input():
    with pytest.raises(ValueError):
        prepare(np.zeros((10, 10, 3), dtype=np.uint8), *NO_NORMALISATION)
    with pytest.raises(ValueError):
        prepare(np.zeros((10, 10), dtype=np.float32), *NO_NORMALISATION)
