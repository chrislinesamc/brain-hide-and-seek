import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from region_search.sections import Roi  # noqa: E402
from region_search.structures import (  # noqa: E402
    MM_PER_PX_X,
    MM_PER_PX_Z,
    SECTION_STEP_MM,
    SECTION_Y0_MM,
    X_LEFT_MM,
    Z_TOP_MM,
    Structure,
    box_hits,
    query_box,
    query_section,
    section_range,
)


def y_of(section):
    return SECTION_Y0_MM + SECTION_STEP_MM * section


def point_at(col, row, section):
    """Point in mm that lands on the centre of thumbnail pixel (col, row) of `section`."""
    return [X_LEFT_MM + (col + 0.5) * MM_PER_PX_X, y_of(section), Z_TOP_MM - (row + 0.5) * MM_PER_PX_Z]


def test_section_range_spans_both_hemispheres():
    points = np.array([point_at(100, 100, 2000), point_at(400, 100, 3000), point_at(100, 100, 2500)])
    structure = Structure("s", points, np.array([True, False, True]))
    assert section_range(structure) == (2000, 3000)


def test_query_box_covers_the_points_on_that_section_and_hemisphere_only():
    points = np.array([
        point_at(100, 200, 3000),
        point_at(120, 230, 3010),  # 0.2 mm away: still on the section
        point_at(50, 50, 3200),  # 4 mm away: not on it
        point_at(400, 200, 3000),  # right hemisphere
    ])
    structure = Structure("s", points, np.array([True, True, True, False]))
    assert query_box(structure, 3000, pad=0) == Roi(100, 200, 21, 31)
    assert query_box(structure, 3000, pad=5) == Roi(95, 195, 31, 41)


def test_query_box_is_clamped_to_the_thumbnail():
    structure = Structure("s", np.array([point_at(2, 430, 3000)]), np.array([True]))
    box = query_box(structure, 3000, pad=8)
    assert (box.x, box.y, box.y + box.h) == (0, 422, 434)


def test_query_box_fails_when_the_structure_misses_the_section():
    structure = Structure("s", np.array([point_at(100, 100, 2000)]), np.array([True]))
    with pytest.raises(ValueError):
        query_box(structure, 3000)


def test_box_hits_when_it_covers_a_point_of_either_hemisphere_on_that_section():
    points = np.array([point_at(100, 200, 3000), point_at(400, 200, 3000), point_at(250, 50, 3200)])
    structure = Structure("s", points, np.array([True, False, True]))
    assert box_hits(structure, 3000, Roi(95, 195, 10, 10))
    assert box_hits(structure, 3000, Roi(395, 195, 10, 10))  # the other hemisphere counts too
    assert not box_hits(structure, 3000, Roi(101, 195, 10, 10))  # misses pixel 100 by one
    assert not box_hits(structure, 3000, Roi(245, 45, 10, 10))  # that point is on section 3200


def test_box_hits_is_none_when_the_structure_has_no_points_on_the_section():
    structure = Structure("s", np.array([point_at(100, 100, 2000)]), np.array([True]))
    assert box_hits(structure, 3000, Roi(0, 0, 500, 434)) is None


def test_boxes_follow_the_stack_transform():
    points = np.array([point_at(100, 200, 3000)])
    structure = Structure("s", points, np.array([True]))
    shift = np.array([[1.0, 0, 30], [0, 1.0, -50]])  # 30 px right, 50 px up
    assert query_box(structure, 3000, pad=0, to_stack=shift, shape=(640, 640)) == Roi(130, 150, 1, 1)
    assert box_hits(structure, 3000, Roi(130, 150, 1, 1), to_stack=shift)
    assert not box_hits(structure, 3000, Roi(100, 200, 1, 1), to_stack=shift)


def test_query_section_is_the_stack_section_nearest_mid_range():
    points = np.array([point_at(100, 100, 2000), point_at(100, 100, 3010)])
    structure = Structure("s", points, np.array([True, True]))
    assert query_section(structure, [1, 2491, 2511, 4000]) == 2511  # mid 2505: 2511 is 6 away, 2491 is 14
