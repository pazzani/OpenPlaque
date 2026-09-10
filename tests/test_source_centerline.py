import numpy as np
import pytest

from openplaque.source_centerline import cumulative_length, point_at_distance, trace_seeded_coronary


class MockImage:
    def __init__(self, shape_zyx, spacing_xyz=(0.7, 0.7, 0.7)):
        self.shape_zyx = tuple(shape_zyx)
        self.spacing_xyz = tuple(spacing_xyz)

    def GetSize(self):
        return (self.shape_zyx[2], self.shape_zyx[1], self.shape_zyx[0])

    def GetSpacing(self):
        return self.spacing_xyz

    def TransformContinuousIndexToPhysicalPoint(self, idx_xyz):
        return tuple(np.asarray(idx_xyz, dtype=float) * np.asarray(self.spacing_xyz, dtype=float))


def make_curved_tube(shape=(100, 80, 80), radius=2.5):
    vol = np.full(shape, 40.0, dtype=np.float32)
    yy, xx = np.ogrid[:shape[1], :shape[2]]
    cx = np.zeros(shape[0]); cy = np.zeros(shape[0])
    for z in range(shape[0]):
        cx[z] = 40 + 12 * np.sin((z - 10) / 80 * np.pi)
        cy[z] = 40 + 8 * np.cos((z - 10) / 80 * np.pi)
    for z in range(10, 91):
        tube = (xx - cx[z]) ** 2 + (yy - cy[z]) ** 2 <= radius ** 2
        vol[z][tube] = 400.0
    return vol, cx, cy


def test_length_helpers():
    pts = np.array([[0., 0., 0.], [3., 0., 0.], [3., 4., 0.]])
    cum = cumulative_length(pts)
    assert np.allclose(cum, [0., 3., 7.])
    assert np.allclose(point_at_distance(pts, cum, 5.), [3., 2., 0.])


def test_seeded_tracer_follows_synthetic_curved_bright_tube():
    vol, cx, cy = make_curved_tube()
    image = MockImage(vol.shape)
    start = (10, int(round(cy[10])), int(round(cx[10])))
    end = (90, int(round(cy[90])), int(round(cx[90])))
    result = trace_seeded_coronary(
        vol,
        image,
        start,
        end,
        crop_margin_mm=10,
        target_spacing_mm=0.7,
    )
    assert result.length_mm > 50
    assert set(result.landmarks_xyz_mm) == {0.0, 10.0, 50.0}
    errors = []
    for z, y, x in result.points_zyx_voxel:
        zi = int(np.clip(round(z), 0, len(cx) - 1))
        errors.append(np.hypot(y - cy[zi], x - cx[zi]))
    assert np.percentile(errors, 95) < 2.0


def test_geometry_mismatch_rejected():
    vol = np.zeros((20, 20, 20), dtype=np.float32)
    image = MockImage((21, 20, 20))
    with pytest.raises(ValueError, match="geometry"):
        trace_seeded_coronary(vol, image, (2, 2, 2), (15, 15, 15))
