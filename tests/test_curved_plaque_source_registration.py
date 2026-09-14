import numpy as np

from openplaque.curved_plaque_source_registration import (
    parallel_transport_frames,
    fit_longitudinal_mapping,
    map_vote_voxels_to_source,
)


def test_parallel_transport_frames_are_orthonormal():
    z=np.linspace(0,20,81)
    p=np.column_stack([2*np.sin(z/12), z, 0.4*np.cos(z/9)])
    t,n,b=parallel_transport_frames(p)
    assert np.allclose(np.linalg.norm(t,axis=1),1,atol=1e-5)
    assert np.allclose(np.linalg.norm(n,axis=1),1,atol=1e-5)
    assert np.allclose(np.linalg.norm(b,axis=1),1,atol=1e-5)
    assert np.max(np.abs((t*n).sum(1))) < 1e-5
    assert np.max(np.abs((t*b).sum(1))) < 1e-5


def test_longitudinal_mapping_recovers_synthetic_window():
    rng=np.random.default_rng(4)
    arc=np.linspace(0,60,121)
    src=350 + 90*np.sin(arc/4.2) + 35*np.cos(arc/1.7)
    stack=rng.normal(0,8,(8,64,256))
    start=54
    for k in range(stack.shape[0]):
        stack[k,31:34,start:start+121] += src[None,:] + rng.normal(0,4,(3,121))
    fit=fit_longitudinal_mapping(arc,src,stack,(0.5,0.5),center_search_px=12)
    assert fit['long_axis']==2
    assert abs(fit['start_px']-start) <= 5
    assert abs(fit['mm_per_long_pixel']-0.5) < 0.08
    assert fit['score'] > 0.75


def test_vote_mapping_straight_centerline():
    arc=np.linspace(0,10,11)
    pts=np.column_stack([arc,np.zeros_like(arc),np.zeros_like(arc)])
    _,n,b=parallel_transport_frames(pts)
    vote=np.zeros((4,9,21),np.uint8)
    vote[0,5,10]=5
    longfit={'long_axis':2,'start_px':0,'n_pixels':21,'long_flip':False,'radial_center_px':4}
    anglefit={'radial_spacing_mm':0.5,'radial_flip':False,'phase_index':0,'angle_direction':1,'angle_period_deg':360.0}
    df=map_vote_voxels_to_source(vote,longfit,anglefit,arc,pts,n,b,(0.5,0.5),threshold=4)
    assert len(df)==1
    assert abs(df.iloc[0]['arc_mm']-5.0) < 1e-6
    assert abs(df.iloc[0]['radial_mm']-0.5) < 1e-6
    assert int(df.iloc[0]['vote'])==5
