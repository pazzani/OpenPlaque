import numpy as np
import pandas as pd

from openplaque.plaque_inflammation_best_estimates_v1 import (
    build_inflammation_estimates,
    build_plaque_estimates,
    build_whole_heart,
    synthetic_self_test,
)


def test_synthetic_self_test():
    assert synthetic_self_test()["ok"] is True


def test_plaque_fusion_reconstructs_ncpv():
    d=pd.DataFrame([
      {"artery":"LAD","voxel_volume_mm3":0.04,"low_attenuation_mm3":2.0,"noncalcified_mm3":3.0,"mixed_intermediate_mm3":4.0,"calcified_mm3":5.0,"strict_nnunet_core_volume_mm3":6.0,"candidate_region_voxels":100,"total_plaque_volume_proxy_mm3":14.0},
      {"artery":"RCA","voxel_volume_mm3":0.04,"low_attenuation_mm3":1.0,"noncalcified_mm3":2.0,"mixed_intermediate_mm3":3.0,"calcified_mm3":4.0,"strict_nnunet_core_volume_mm3":5.0,"candidate_region_voxels":50,"total_plaque_volume_proxy_mm3":10.0},
      {"artery":"LCX","voxel_volume_mm3":0.04,"low_attenuation_mm3":1.0,"noncalcified_mm3":1.0,"mixed_intermediate_mm3":1.0,"calcified_mm3":1.0,"strict_nnunet_core_volume_mm3":2.0,"candidate_region_voxels":25,"total_plaque_volume_proxy_mm3":4.0},
      {"artery":"Left main","low_attenuation_mm3":0.0,"noncalcified_mm3":0.0,"mixed_intermediate_mm3":0.0,"calcified_mm3":54.0,"strict_nnunet_core_volume_mm3":0.0,"candidate_region_voxels":0,"total_plaque_volume_proxy_mm3":54.0},
    ])
    p=build_plaque_estimates(d)
    lad=p[p.vessel=="LAD"].iloc[0]
    assert np.isclose(lad.ncpv_best_estimate_mm3,9.0)
    assert np.isclose(lad.tpv_candidate_envelope_upper_mm3,10.0)
    lm=p[p.vessel=="LM"].iloc[0]
    assert np.isnan(lm.tpv_candidate_envelope_upper_mm3)


def test_whole_major_vessel_not_literal_whole_heart():
    d=pd.DataFrame([
      {"artery":"LAD","voxel_volume_mm3":0.04,"low_attenuation_mm3":2.0,"noncalcified_mm3":3.0,"mixed_intermediate_mm3":4.0,"calcified_mm3":5.0,"strict_nnunet_core_volume_mm3":6.0,"candidate_region_voxels":100,"total_plaque_volume_proxy_mm3":14.0},
      {"artery":"RCA","voxel_volume_mm3":0.04,"low_attenuation_mm3":1.0,"noncalcified_mm3":2.0,"mixed_intermediate_mm3":3.0,"calcified_mm3":4.0,"strict_nnunet_core_volume_mm3":5.0,"candidate_region_voxels":50,"total_plaque_volume_proxy_mm3":10.0},
      {"artery":"LCX","voxel_volume_mm3":0.04,"low_attenuation_mm3":1.0,"noncalcified_mm3":1.0,"mixed_intermediate_mm3":1.0,"calcified_mm3":1.0,"strict_nnunet_core_volume_mm3":2.0,"candidate_region_voxels":25,"total_plaque_volume_proxy_mm3":4.0},
      {"artery":"Left main","low_attenuation_mm3":0.0,"noncalcified_mm3":0.0,"mixed_intermediate_mm3":0.0,"calcified_mm3":54.0,"strict_nnunet_core_volume_mm3":0.0,"candidate_region_voxels":0,"total_plaque_volume_proxy_mm3":54.0},
    ])
    w=build_whole_heart(build_plaque_estimates(d)).iloc[0]
    assert w.whole_coronary_tree_equivalence=="NO"
    assert w.confirm2_tpv_stage_best_estimate==">0-250 mm3"


def test_inflammation_preserves_caristo_boundary():
    rca=pd.DataFrame([
      {"mean_hu":-90.0,"fat_voxels":100},
      {"mean_hu":-80.0,"fat_voxels":300},
    ])
    lad=pd.DataFrame([{
      "segment":"frozen_LAD","pcat_mean_hu":-81.2,
      "frozen_arc_start_mm":1.3,"frozen_arc_end_mm":23.8,
      "segment_length_mm":22.5,"fat_voxels":20000
    }])
    lcx=pd.DataFrame([
      {"vessel":"C6","pcat_mean_hu":-86.4,"segment_arc_start_mm":1.0,"segment_arc_end_mm":6.5,"segment_length_mm":5.5,"fat_voxels":6000},
      {"vessel":"C7","pcat_mean_hu":-83.9,"segment_arc_start_mm":1.0,"segment_arc_end_mm":17.0,"segment_length_mm":16.0,"fat_voxels":9000},
    ])
    out=build_inflammation_estimates(rca,lad,lcx)
    assert np.isclose(out[out.vessel=="RCA"].iloc[0].pcat_mean_hu_best_estimate,-82.5)
    assert out[out.vessel=="RCA"].iloc[0].fai_score=="NA"
    assert np.isclose(out[out.vessel=="LAD"].iloc[0].segment_coverage_fraction,22.5/40.0)
    assert out[out.vessel=="LM"].iloc[0].caristo_comparability=="not standardized"
