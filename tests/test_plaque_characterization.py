import numpy as np
from types import SimpleNamespace
from openplaque.plaque_characterization import classify_plaque_voxels, summarize_plaque_types, summarize_all_vessels

class FakeImage:
    def GetSpacing(self):
        return (1.0, 1.0, 2.0)

def test_classify_plaque_voxels():
    vol = np.array([[[10, 50, 200, 500, 0]]], dtype=float)
    mask = np.array([[[2, 2, 2, 2, 0]]], dtype=np.uint8)
    cls = classify_plaque_voxels(vol, mask)
    assert cls.tolist() == [[[1, 2, 3, 4, 0]]]

def test_summarize_plaque_types():
    vol = np.zeros((4, 4, 4), dtype=float)
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1,1,1] = 2; vol[1,1,1] = 10
    mask[1,1,2] = 2; vol[1,1,2] = 100
    mask[1,2,1] = 2; vol[1,2,1] = 200
    mask[1,2,2] = 2; vol[1,2,2] = 500
    summary, lesions, class_map = summarize_plaque_types('LAD', vol, mask, (1,1,1))
    assert summary.total_plaque_voxels == 4
    assert summary.low_attenuation_voxels == 1
    assert summary.noncalcified_voxels == 1
    assert summary.mixed_intermediate_voxels == 1
    assert summary.calcified_voxels == 1
    assert len(lesions) == 1

def test_summarize_all_vessels():
    vol = np.zeros((3,3,3), dtype=float)
    mask = np.zeros((3,3,3), dtype=np.uint8)
    mask[1,1,1] = 2
    report = SimpleNamespace(name='RCA', volume=vol, mask_image=FakeImage())
    refinement = SimpleNamespace(refined_mask=mask)
    summaries, lesions, maps = summarize_all_vessels([report], {'RCA': refinement})
    assert len(summaries) == 1
    assert 'RCA' in maps
