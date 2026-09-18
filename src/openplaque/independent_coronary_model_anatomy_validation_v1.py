from __future__ import annotations

import json, math, zipfile
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi

BASELINE="0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM="independent-coronary-model-anatomy-validation-v1.0"
OUTPUT_DIRNAME="Independent_Coronary_Model_Anatomy_Validation_v1"
SOURCE_CACHE=Path("Cache/Secondary_3D_Vesselness_Topology_v1")
LAD_PATH=Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
RCA_PATH=Path("Cache/Source_Volume_Coronary_Centerlines/RCA_source_centerline.csv")
C6_PATH=Path("Joint_Three_Vessel_Template_Classifier_v1/candidate_04_source_path.csv")
C7_PATH=Path("LCX_Distal_Reacquisition_v1_fixed/C7_extended_path.csv")
AORTA_MASK=Path("TotalSegmentator_Cardiovascular_Cache_v1/heartchambers_highres/aorta.nii.gz")
MASTER=Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
EXPECTED_MASTER_STATUS="CORONARY_ANATOMY_BASELINE_V2_FROZEN"

# Prespecified before viewing the external model output on this scan.
PATH_TOLERANCE_MM=1.25
AORTA_CONTACT_TOLERANCE_MM=1.50
MIN_RCA_SUPPORT=0.70
MIN_LAD_SUPPORT=0.70
MIN_C6_SUPPORT=0.60
MIN_C7_SUPPORT=0.50

STATUS_POSITIVE="INDEPENDENT_MODEL_LEFT_CORONARY_AORTIC_CONTINUITY_POSITIVE"
STATUS_NEGATIVE="INDEPENDENT_MODEL_NO_LEFT_CORONARY_AORTIC_CONTINUITY"
STATUS_CONTROL_FAIL="INDEPENDENT_MODEL_RCA_CONTROL_FAILED"

def _req(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(str(p))
    return p

def _read_json(p): return json.loads(_req(p).read_text(encoding="utf-8"))
def _write_json(p,obj): Path(p).write_text(json.dumps(obj,indent=2,default=str,allow_nan=True),encoding="utf-8")

def _arc(p):
    p=np.asarray(p,float)
    return np.zeros(len(p)) if len(p)<=1 else np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]

def _resample_polyline(p,step_mm=.5):
    p=np.asarray(p,float); a=_arc(p)
    if len(p)<2 or a[-1]<=0: return p.copy()
    q=np.arange(0.,a[-1]+1e-9,float(step_mm))
    if q[-1]<a[-1]-1e-6: q=np.r_[q,a[-1]]
    return np.column_stack([np.interp(q,a,p[:,k]) for k in range(3)])

def _load_lps_csv(p):
    d=pd.read_csv(_req(p))
    for cols in (("lps_x_mm","lps_y_mm","lps_z_mm"),("x_mm","y_mm","z_mm")):
        if all(c in d.columns for c in cols): return d[list(cols)].to_numpy(float)
    raise ValueError(f"No supported LPS columns in {p}: {list(d.columns)}")

class SourceGeometry:
    def __init__(self,meta):
        self.spacing_zyx=np.asarray(meta["spacing_zyx"],float)
        self.spacing_xyz=self.spacing_zyx[::-1]
        self.origin=np.asarray(meta["positions_lps_mm"][0],float)
        iop=np.asarray(meta["image_orientation_patient"],float)
        row,col=iop[:3],iop[3:]; slc=np.cross(row,col)
        self.direction=np.array([[row[0],col[0],slc[0]],[row[1],col[1],slc[1]],[row[2],col[2],slc[2]]],float)
        self.inv_direction=np.linalg.inv(self.direction)
    def xyz_to_zyx_float(self,pts):
        pts=np.atleast_2d(np.asarray(pts,float))
        xyz=((pts-self.origin)@self.inv_direction.T)/self.spacing_xyz
        return xyz[:,::-1]

def _load_source(root):
    c=Path(root)/SOURCE_CACHE
    src=np.load(_req(c/"series7_int16.npy"),mmap_mode="r")
    meta=_read_json(c/"series7_int16.json")
    return SourceGeometry(meta),src,meta

def write_source_nifti(drive_root="/content/drive/MyDrive/OpenPlaque",out_path=None):
    root=Path(drive_root); geom,src,_=_load_source(root)
    out=Path(out_path) if out_path else root/OUTPUT_DIRNAME/"input"/"ucla_series7.img.nii.gz"
    out.parent.mkdir(parents=True,exist_ok=True)
    img=sitk.GetImageFromArray(np.asarray(src))
    img.SetSpacing(tuple(float(v) for v in geom.spacing_xyz))
    img.SetOrigin(tuple(float(v) for v in geom.origin))
    img.SetDirection(tuple(float(v) for v in geom.direction.ravel()))
    sitk.WriteImage(img,str(out),True)
    return str(out)

def _source_ref(geom,shape):
    r=sitk.Image(tuple(int(v) for v in shape[::-1]),sitk.sitkUInt8)
    r.SetSpacing(tuple(float(v) for v in geom.spacing_xyz))
    r.SetOrigin(tuple(float(v) for v in geom.origin))
    r.SetDirection(tuple(float(v) for v in geom.direction.ravel()))
    return r

def _mask_on_source(path,geom,shape):
    img=sitk.ReadImage(str(_req(path))); ref=_source_ref(geom,shape)
    same=(tuple(img.GetSize())==tuple(ref.GetSize()) and np.allclose(img.GetSpacing(),ref.GetSpacing(),atol=1e-4)
          and np.allclose(img.GetOrigin(),ref.GetOrigin(),atol=1e-4) and np.allclose(img.GetDirection(),ref.GetDirection(),atol=1e-4))
    if not same: img=sitk.Resample(img,ref,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
    a=sitk.GetArrayFromImage(img)>0
    if tuple(a.shape)!=tuple(shape): raise RuntimeError(f"Mask shape {a.shape} != source {shape}")
    return a

def _distance_labels(mask,spacing):
    labels,n=ndi.label(mask,structure=np.ones((3,3,3),np.uint8))
    dist,inds=ndi.distance_transform_edt(~mask,sampling=spacing,return_indices=True)
    return dist,inds,labels,int(n)

def _sample_nearest(points,geom,dist,inds,labels):
    z=np.rint(geom.xyz_to_zyx_float(points)).astype(int); shape=np.asarray(dist.shape)
    valid=np.all((z>=0)&(z<shape),axis=1); d=np.full(len(z),np.inf); lab=np.zeros(len(z),int)
    if valid.any():
        zv=z[valid]; d[valid]=dist[tuple(zv.T)]
        near=np.column_stack([inds[k][tuple(zv.T)] for k in range(3)]).astype(int)
        lab[valid]=labels[tuple(near.T)]
    return d,lab,valid

def _path_support(points,geom,dist,inds,labels,tol_mm=PATH_TOLERANCE_MM):
    p=_resample_polyline(points,.5); d,lab,valid=_sample_nearest(p,geom,dist,inds,labels)
    ok=valid&(d<=tol_mm)&(lab>0); labs=lab[ok]
    ids,cnts=np.unique(labs,return_counts=True) if len(labs) else (np.array([],int),np.array([],int))
    dominant=int(ids[np.argmax(cnts)]) if len(ids) else 0
    return {"n_samples":int(len(p)),"support_fraction":float(ok.mean()) if len(p) else 0.,
            "median_distance_mm":float(np.median(d[np.isfinite(d)])) if np.isfinite(d).any() else math.inf,
            "p90_distance_mm":float(np.percentile(d[np.isfinite(d)],90)) if np.isfinite(d).any() else math.inf,
            "dominant_component":dominant,"supported_component_ids":sorted(int(x) for x in set(labs.tolist())),
            "_points":p,"_dist":d,"_labels":lab}

def _aorta_contacts(labels,aorta,spacing,tol=AORTA_CONTACT_TOLERANCE_MM):
    da=ndi.distance_transform_edt(~aorta,sampling=spacing); touch=(labels>0)&(da<=tol)
    ids,cnts=np.unique(labels[touch],return_counts=True)
    return {int(i):int(c) for i,c in zip(ids,cnts) if int(i)>0}

def _shared(a,b): return sorted(set(a["supported_component_ids"])&set(b["supported_component_ids"]))

def _plot_support(rows,out):
    d=pd.DataFrame(rows); fig,ax=plt.subplots(figsize=(8,4.8)); ax.bar(d.path,d.support_fraction)
    ax.set_ylim(0,1.05); ax.set_ylabel("Fraction within 1.25 mm"); ax.set_title("Independent model support for frozen paths")
    for i,v in enumerate(d.support_fraction): ax.text(i,min(1.02,v+.03),f"{v:.2f}",ha="center")
    fig.tight_layout(); fig.savefig(out,dpi=170); plt.close(fig)

def _plot_topology(mask,geom,paths,contacts,out):
    pts=np.argwhere(mask)
    if len(pts)>25000: pts=pts[np.linspace(0,len(pts)-1,25000).astype(int)]
    xyz=pts[:,::-1]*geom.spacing_xyz; lps=geom.origin+xyz@geom.direction.T
    fig=plt.figure(figsize=(10,8)); ax=fig.add_subplot(111,projection="3d")
    if len(lps): ax.scatter(lps[:,0],lps[:,1],lps[:,2],s=.5,alpha=.1,label="ImageCAS-X lumen")
    for n,p in paths.items(): ax.plot(p[:,0],p[:,1],p[:,2],lw=2.2,label=n)
    ax.set_title(f"Independent lumen topology; aorta-touching components={sorted(contacts)}"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out,dpi=170); plt.close(fig)

def synthetic_self_test():
    m=np.zeros((20,20,20),bool); m[10,2:18,10]=1
    dist,inds,lab,n=_distance_labels(m,(1,1,1)); assert n==1 and lab[10,10,10]==1 and dist[10,10,10]==0
    a=np.zeros_like(m); a[10,1:4,9:12]=1; assert 1 in _aorta_contacts(lab,a,(1,1,1),1.5)
    return {"ok":True,"components":n}

def run(prediction_mask,drive_root="/content/drive/MyDrive/OpenPlaque",output_dir=None,
        model_name="ImageCAS-X CAS-Net",model_record="Zenodo 21887809"):
    root=Path(drive_root); out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME; out.mkdir(parents=True,exist_ok=True)
    _write_json(out/"run_state.json",{"status":"STARTED","algorithm":ALGORITHM,"baseline":BASELINE})
    master=_read_json(root/MASTER)
    if master.get("status")!=EXPECTED_MASTER_STATUS: raise RuntimeError(f"Frozen master prerequisite failed: {master.get('status')}")
    geom,src,_=_load_source(root); shape=tuple(int(v) for v in src.shape)
    pred=_mask_on_source(prediction_mask,geom,shape); aorta=_mask_on_source(root/AORTA_MASK,geom,shape)
    if not pred.any(): raise RuntimeError("Independent-model prediction is empty")
    dist,inds,labels,ncomp=_distance_labels(pred,geom.spacing_zyx); contacts=_aorta_contacts(labels,aorta,geom.spacing_zyx); contact_ids=set(contacts)
    paths={"LAD":_load_lps_csv(root/LAD_PATH),"RCA":_load_lps_csv(root/RCA_PATH),"C6":_load_lps_csv(root/C6_PATH),"C7":_load_lps_csv(root/C7_PATH)}
    res={k:_path_support(v,geom,dist,inds,labels) for k,v in paths.items()}
    shared_lad_c6=_shared(res["LAD"],res["C6"]); shared_c6_c7=_shared(res["C6"],res["C7"])
    left_aorta=sorted(set(shared_lad_c6)&contact_ids); rca_aorta=sorted(set(res["RCA"]["supported_component_ids"])&contact_ids)
    gates={"master_still_frozen":True,"RCA_support_control":res["RCA"]["support_fraction"]>=MIN_RCA_SUPPORT,
           "RCA_component_reaches_aorta_control":bool(rca_aorta),"LAD_support":res["LAD"]["support_fraction"]>=MIN_LAD_SUPPORT,
           "C6_support":res["C6"]["support_fraction"]>=MIN_C6_SUPPORT,"LAD_C6_share_model_component":bool(shared_lad_c6),
           "shared_LAD_C6_component_reaches_aorta":bool(left_aorta)}
    control=bool(gates["RCA_support_control"] and gates["RCA_component_reaches_aorta_control"])
    left=bool(gates["LAD_support"] and gates["C6_support"] and gates["LAD_C6_share_model_component"] and gates["shared_LAD_C6_component_reaches_aorta"])
    status=STATUS_CONTROL_FAIL if not control else (STATUS_POSITIVE if left else STATUS_NEGATIVE)

    rows=[]; point=[]
    for name,r in res.items():
        rows.append({"path":name,"support_fraction":r["support_fraction"],"median_distance_mm":r["median_distance_mm"],
                     "p90_distance_mm":r["p90_distance_mm"],"dominant_component":r["dominant_component"],
                     "aorta_touching_supported_component":bool(set(r["supported_component_ids"])&contact_ids)})
        p=r["_points"]; point.append(pd.DataFrame({"path":name,"sample_index":np.arange(len(p)),"arc_mm":_arc(p),
                    "lps_x_mm":p[:,0],"lps_y_mm":p[:,1],"lps_z_mm":p[:,2],"distance_to_model_lumen_mm":r["_dist"],
                    "nearest_model_component":r["_labels"],"within_tolerance":r["_dist"]<=PATH_TOLERANCE_MM}))
    pd.DataFrame(rows).to_csv(out/"path_support_summary.csv",index=False)
    pd.concat(point,ignore_index=True).to_csv(out/"path_point_diagnostics.csv",index=False)
    pd.DataFrame([{"gate":k,"pass":bool(v)} for k,v in gates.items()]).to_csv(out/"prespecified_gates.csv",index=False)
    pd.DataFrame([{"component":k,"contact_voxels":v} for k,v in sorted(contacts.items())]).to_csv(out/"aorta_contact_components.csv",index=False)
    _plot_support(rows,out/"01_independent_model_path_support.png"); _plot_topology(pred,geom,paths,contacts,out/"02_independent_model_topology.png")

    compact={k:{kk:vv for kk,vv in r.items() if not kk.startswith("_")} for k,r in res.items()}
    decision={"status":status,"model_control_pass":control,"independent_left_coronary_aortic_continuity_positive":bool(control and left),
              "shared_LAD_C6_components":shared_lad_c6,"shared_C6_C7_components":shared_c6_c7,
              "left_components_touching_aorta":left_aorta,"RCA_components_touching_aorta":rca_aorta,
              "clinical_LM_identity_established":False,"clinical_LCX_OM_identity_established":False,"master_anatomy_modified":False}
    summary={"status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE,
             "external_model":{"name":model_name,"record":model_record,"role":"independent binary coronary-lumen observer"},
             "source_shape_zyx":list(shape),"source_spacing_zyx_mm":[float(v) for v in geom.spacing_zyx],
             "prediction_voxels":int(pred.sum()),"prediction_connected_components":ncomp,
             "path_tolerance_mm":PATH_TOLERANCE_MM,"aorta_contact_tolerance_mm":AORTA_CONTACT_TOLERANCE_MM,
             "prespecified_thresholds":{"MIN_RCA_SUPPORT":MIN_RCA_SUPPORT,"MIN_LAD_SUPPORT":MIN_LAD_SUPPORT,
                 "MIN_C6_SUPPORT":MIN_C6_SUPPORT,"MIN_C7_SUPPORT_descriptive_only":MIN_C7_SUPPORT},
             "paths":compact,"gates":gates,"decision":decision,
             "scientific_boundary":"Independent learned-lumen evidence only. Frozen master remains unchanged; clinical LM/LCX/OM identity is not promoted."}
    _write_json(out/"summary.json",summary); _write_json(out/"decision.json",decision)
    _write_json(out/"input_provenance.json",{"prediction_mask":str(prediction_mask),"source_cache":str(root/SOURCE_CACHE),
        "aorta_mask":str(root/AORTA_MASK),"LAD":str(root/LAD_PATH),"RCA":str(root/RCA_PATH),"C6":str(root/C6_PATH),"C7":str(root/C7_PATH),"master":str(root/MASTER)})
    report=out/"OPENPLAQUE_INDEPENDENT_CORONARY_MODEL_ANATOMY_VALIDATION_V1_REPORT.html"
    report.write_text("<html><body><h1>Independent Coronary Model Anatomy Validation v1</h1>"
        f"<p><b>Status:</b> {status}</p><p><b>Model:</b> {model_name} ({model_record})</p>"
        f"<p>RCA control pass: {control}; independent left aortic continuity positive: {bool(control and left)}.</p>"
        "<p>External binary-lumen observer only. Frozen master unchanged; no clinical LM/LCX/OM promotion.</p>"
        "<h2>Path support</h2>"+pd.DataFrame(rows).to_html(index=False)+"<h2>Gates</h2>"
        +pd.DataFrame([{"gate":k,"pass":v} for k,v in gates.items()]).to_html(index=False)
        +"<h2>QC</h2><img src='01_independent_model_path_support.png' width='850'><br><img src='02_independent_model_topology.png' width='850'></body></html>",encoding="utf-8")
    _write_json(out/"run_state.json",{"status":"COMPLETE","algorithm":ALGORITHM,"result_status":status})
    archive=out/"OPENPLAQUE_INDEPENDENT_CORONARY_MODEL_ANATOMY_VALIDATION_V1_RESULTS.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as zf:
        for p in out.iterdir():
            if p.is_file() and p!=archive: zf.write(p,p.name)
    return summary
