import numpy as np

from openplaque.lad_source_space_pcat_feasibility_v1 import (
    BASELINE,
    STATUS_PASS,
    PRIMARY_WALL_MARGIN_MM,
    FROZEN_SEGMENT_RANGE_MM,
    DISTAL_SEGMENT_RANGE_MM,
    _join,
    synthetic_self_test,
)


def test_constants_and_self_test():
    assert BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert STATUS_PASS == "LAD_SOURCE_SPACE_PCAT_FEASIBILITY_COMPLETE"
    assert PRIMARY_WALL_MARGIN_MM == 0.75
    assert FROZEN_SEGMENT_RANGE_MM == (1.0, 24.0)
    assert DISTAL_SEGMENT_RANGE_MM == (-14.0, -1.0)
    assert synthetic_self_test()["ok"] is True


def test_join_distal_extension():
    frozen = np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
    extension = np.array([[-2.,0.,0.],[-1.,0.,0.],[0.,0.,0.]])
    combined, ext_len, gap = _join(frozen, extension)
    assert np.isclose(ext_len, 2.0)
    assert np.isclose(gap, 0.0)
    assert np.allclose(combined[0], [-2.,0.,0.])
    assert np.allclose(combined[-1], [2.,0.,0.])
