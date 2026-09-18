import numpy as np

from openplaque.rca_expert_outer_wall_annotation_pack_v1 import (
    BASELINE,
    STATUS,
    ANNOTATION_STEP_MM,
    PLANE_PIXEL_MM,
    _lumen_mask_from_boundary,
    _validate_outer_mask,
    synthetic_self_test,
)


def test_constants_and_self_test():
    assert BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert STATUS == "RCA_EXPERT_OUTER_WALL_ANNOTATION_PACK_READY"
    assert ANNOTATION_STEP_MM == 1.0
    assert PLANE_PIXEL_MM == 0.15
    assert synthetic_self_test()["ok"] is True


def test_circular_lumen_mask_area():
    q = np.arange(-2.0, 2.0 + 1e-9, 0.1)
    yy, xx = np.meshgrid(q, q, indexing="ij")
    theta = np.linspace(0, 2*np.pi, 72, endpoint=False)
    radii = np.full(72, 1.25)
    m = _lumen_mask_from_boundary(xx, yy, theta, radii)
    area = m.sum() * 0.1 * 0.1
    assert abs(area - np.pi*1.25**2) < 0.20


def test_expert_outer_mask_validation_and_components():
    q = np.arange(-2.0, 2.0 + 1e-9, 0.1)
    yy, xx = np.meshgrid(q, q, indexing="ij")
    theta = np.linspace(0, 2*np.pi, 72, endpoint=False)
    radii = np.full(72, 0.8)
    lum = _lumen_mask_from_boundary(xx, yy, theta, radii).astype(bool)
    outer = (np.sqrt(xx*xx + yy*yy) <= 1.3)

    hu = np.full_like(xx, 60.0, dtype=float)
    hu[(np.sqrt(xx*xx + yy*yy) > 1.0) & outer] = 10.0

    v = _validate_outer_mask(
        hu[None, ...],
        lum[None, ...],
        outer[None, ...],
        pixel_mm=0.1,
        step_mm=1.0,
    )
    assert bool(v.outer_contains_lumen.iloc[0])
    assert v.wall_volume_mm3.iloc[0] > 0
    assert v.low_attenuation_mm3.iloc[0] > 0
    assert v.noncalcified_mm3.iloc[0] > 0


def test_empty_template_is_not_valid_outer_contour():
    hu = np.zeros((1, 10, 10), float)
    lum = np.zeros((1, 10, 10), np.uint8)
    lum[:, 4:6, 4:6] = 1
    outer = np.zeros_like(lum)
    v = _validate_outer_mask(hu, lum, outer)
    assert not bool(v.outer_contains_lumen.iloc[0])
    assert v.outer_pixels.iloc[0] == 0
