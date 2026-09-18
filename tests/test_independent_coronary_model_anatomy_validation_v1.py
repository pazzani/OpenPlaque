import numpy as np

from openplaque.independent_coronary_model_anatomy_validation_v1 import (
    _aorta_contacts,
    _arc,
    _distance_labels,
    _resample_polyline,
    synthetic_self_test,
)

def test_synthetic_self_test():
    out = synthetic_self_test()
    assert out["ok"] is True
    assert out["components"] == 1

def test_resample_polyline():
    p=np.array([[0.,0.,0.],[2.,0.,0.]])
    q=_resample_polyline(p,.5)
    assert len(q)==5
    assert np.isclose(_arc(q)[-1],2.0)

def test_aorta_contact_is_component_specific():
    m=np.zeros((20,20,20),bool)
    m[10,2:8,5]=1
    m[10,12:18,15]=1
    _,_,labels,n=_distance_labels(m,(1.,1.,1.))
    assert n==2
    a=np.zeros_like(m)
    a[10,1:4,4:7]=1
    contacts=_aorta_contacts(labels,a,(1.,1.,1.),1.5)
    assert len(contacts)==1
    assert min(contacts)==labels[10,2,5]
