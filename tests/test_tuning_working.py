import numpy as np
from dataclasses import dataclass

from openplaque.tuning import tune_boundary_parameters, best_parameters, best_rows_by_vessel, apply_refinement_with_params

class FakeImage:
    def GetSpacing(self):
        return (1.0, 1.0, 1.0)

@dataclass
class FakeReport:
    name: str
    volume: np.ndarray
    mask: np.ndarray
    mask_image: object
    @property
    def tpv_mm3(self):
        return float((self.mask == 2).sum())


def test_tuning_smoke():
    volume = np.zeros((20, 20, 20), dtype=np.float32)
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[8:13, 8:13, 8:13] = 2
    mask[7:14, 7:14, 7] = 1
    volume[mask == 2] = 120
    report = FakeReport('LAD', volume, mask, FakeImage())
    grid = {
        'min_component_voxels': [1, 10],
        'lumen_distance_voxels': [0, 1],
        'high_hu_threshold': [None],
        'low_hu_threshold': [None],
        'erode_core': [False],
        'erosion_iterations': [1],
    }
    df = tune_boundary_parameters([report], grid)
    assert not df.empty
    assert 'score' in df.columns
    params = best_parameters(df)
    assert isinstance(params, dict)
    best = best_rows_by_vessel(df)
    assert len(best) == 1
    refs = apply_refinement_with_params([report], params)
    assert 'LAD' in refs
