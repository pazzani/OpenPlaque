import numpy as np
from types import SimpleNamespace
from openplaque.tuning import tune_boundary_parameters

class DummyImage:
    def GetSpacing(self):
        return (1.0, 1.0, 1.0)

def test_tuning_smoke():
    mask = np.zeros((8, 16, 16), dtype=np.uint8)
    vol = np.zeros((8, 16, 16), dtype=np.float32)
    mask[:, 5:11, 5:11] = 1
    mask[:, 7:10, 7:10] = 2
    mask[0, 0, 0] = 2
    report = SimpleNamespace(name='LAD', volume=vol, mask=mask, mask_image=DummyImage(), tpv_mm3=float(np.sum(mask==2)), plaque_voxels=int(np.sum(mask==2)))
    res = tune_boundary_parameters([report], parameter_grid={
        'min_component_voxels': [1, 5],
        'lumen_distance_voxels': [0, 1],
        'high_hu_threshold': [None],
        'low_hu_threshold': [None],
        'erode_core': [False],
        'erosion_iterations': [1],
    })
    assert not res.empty
    assert 'LAD' in set(res['vessel'])
    assert 'min_component_voxels' in res.columns
