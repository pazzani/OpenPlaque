from pathlib import Path

from openplaque import lcx_consensus_trunk_av_groove as m

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"

def test_frozen_baseline_and_algorithm():
    assert m.BASELINE == BASELINE
    assert m.ALGORITHM == "lcx-consensus-trunk-av-groove-v1.0"

def test_expected_heart_chamber_inputs():
    assert str(m.LA).endswith("heartchambers_highres/heart_atrium_left.nii.gz")
    assert str(m.LV).endswith("heartchambers_highres/heart_ventricle_left.nii.gz")
    assert str(m.RA).endswith("heartchambers_highres/heart_atrium_right.nii.gz")
    assert str(m.RV).endswith("heartchambers_highres/heart_ventricle_right.nii.gz")

def test_five_prior_leaf_paths_are_explicit():
    assert len(m.LEAF_FILES) == 5
    assert str(m.LEAF_FILES[0]).endswith("candidate_01_source_path.csv")
    assert str(m.LEAF_FILES[-1]).endswith("candidate_05_source_path.csv")

def test_synthetic_consensus_prefix():
    r = m.synthetic_av_groove_self_test()
    assert r["ok"] is True
    assert 14.5 <= r["consensus_mm"] <= 15.5

def test_statuses_remain_conservative():
    assert "REQUIRES_VISUAL_QC" in m.STATUS_CANDIDATE
    assert "AMBIGUOUS" in m.STATUS_AMBIG
    assert "CONTROL_FAILED" in m.STATUS_CONTROL_FAIL

def test_no_stale_lcx_source_centerline_dependency():
    source = Path(m.__file__).read_text(encoding="utf-8")
    assert "LCX_source_centerline.csv" not in source
