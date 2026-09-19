from __future__ import annotations

import json, zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "series6-native-all-component-gap-analysis-v1.0"
OUTPUT_DIRNAME = "Series6_Native_All_Component_Gap_Analysis_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
LAD_PATH = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
RCA_PATH = Path("Cache/Source_Volume_Coronary_Centerlines/RCA_source_centerline.csv")
C6_PATH = Path("Joint_Three_Vessel_Template_Classifier_v1/candidate_04_source_path.csv")
C7_PATH = Path("LCX_Distal_Reacquisition_v1_fixed/C7_extended_path.csv")
SOURCE_AORTA = Path("TotalSegmentator_Cardiovascular_Cache_v1/heartchambers_highres/aorta.nii.gz")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
EXPECTED_MASTER_STATUS = "CORONARY_ANATOMY_BASELINE_V2_FROZEN"

SERIES6_FOLDER_NAME = "3228"
ASSOCIATION_RADIUS_MM = 4.0
MIN_ASSOCIATED_VOXELS = 20
SHORT_GAP_MM = 5.0
GAP_SAMPLE_STEP_MM = 0.25
CONTRAST_HU_THRESHOLD = 150.0
CONTRAST_FRACTION_THRESHOLD = 0.60
MAX_ROOT_TRANSLATION_MM = 6.0

STATUS_SHORT_CONTRAST = "SERIES6_LEFT_REGION_SHORT_GAP_CONTRAST_SUPPORTED"
STATUS_SHORT_NO_CONTRAST = "SERIES6_LEFT_REGION_SHORT_GAP_NO_CONTRAST_SUPPORT"
STATUS_NO_SHORT = "SERIES6_LEFT_REGION_NO_SHORT_GAP"
STATUS_LOCALIZATION_FAIL = "SERIES6_LEFT_REGION_LOCALIZATION_FAILED"
STATUS_REG_FAIL = "SERIES6_ROOT_REGISTRATION_FAILED"


def _req(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(str(p))
    return p


def _read_json(p): return json.loads(_req(p).read_text())


def _write_json(p,x): Path(p).write_text(json.dumps(x,indent=2,default=str,allow_nan=True))


def _safe_float(x,default=np.nan):
    try: return float(x)
    except Exception: return float(default)


def _load_path(p):
    d=pd.read_csv(_req(p))
    for cols in (("lps_x_mm","lps_y_mm","lps_z_mm"),("x_mm","y_mm","z_mm")):
        if all(c in d.columns for c in cols): return d[list(cols)].to_numpy(float)
    raise ValueError(f"No supported LPS columns in {p}")


def _arc(p):
    p=np.asarray(p,float)
    return np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))] if len(p)>1 else np.zeros(len(p))


def _resample_path(p,step=.5):
    p=np.asarray(p,float); a=_arc(p)
    if len(p)<2 or a[-1]<=0: return p.copy()
    q=np.arange(0,a[-1]+1e-9,step)
    if q[-1]<a[-1]-1e-6: q=np.r_[q,a[-1]]
    return np.column_stack([np.interp(q,a,p[:,k]) for k in range(3)])


@dataclass
class Geometry:
    spacing_xyz: np.ndarray
    origin: np.ndarray
    direction: np.ndarray

    @property
    def spacing_zyx(self): return self.spacing_xyz[::-1]

    def xyz_to_zyx(self,pts):
        pts=np.atleast_2d(np.asarray(pts,float))
        xyz=((pts-self.origin)@np.linalg.inv(self.direction).T)/self.spacing_xyz
        return xyz[:,::-1]

    def zyx_to_xyz(self,zyx):
        zyx=np.atleast_2d(np.asarray(zyx,float))
        return self.origin+(zyx[:,::-1]*self.spacing_xyz)@self.direction.T


def _same_geometry(a,b):
    return (tuple(a.GetSize())==tuple(b.GetSize())
            and np.allclose(a.GetSpacing(),b.GetSpacing(),atol=1e-4)
            and np.allclose(a.GetOrigin(),b.GetOrigin(),atol=1e-3)
            and np.allclose(a.GetDirection(),b.GetDirection(),atol=1e-4))


def _source_reference(root):
    cache=Path(root)/SOURCE_CACHE
    arr=np.load(_req(cache/"series7_int16.npy"),mmap_mode="r")
    m=_read_json(cache/"series7_int16.json")
    sp=np.asarray(m["spacing_zyx"],float)[::-1]
    iop=np.asarray(m["image_orientation_patient"],float)
    D=np.column_stack([iop[:3],iop[3:],np.cross(iop[:3],iop[3:])])
    g=Geometry(sp,np.asarray(m["positions_lps_mm"][0],float),D)
    img=sitk.GetImageFromArray(np.asarray(arr))
    img.SetSpacing(tuple(g.spacing_xyz)); img.SetOrigin(tuple(g.origin)); img.SetDirection(tuple(D.ravel()))
    return img,np.asarray(arr),g


def _source_aorta(root,ref):
    a=sitk.ReadImage(str(_req(Path(root)/SOURCE_AORTA)))
    if not _same_geometry(a,ref):
        a=sitk.Resample(a,ref,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
    return sitk.GetArrayFromImage(a)>0


def _natural_key(p):
    import re
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)",Path(p).name)]


def load_series6(folder):
    rec=[]
    for p in sorted(Path(folder).iterdir(),key=_natural_key):
        if not p.is_file(): continue
        try:
            ds=pydicom.dcmread(str(p),force=True)
            if hasattr(ds,"PixelData") and hasattr(ds,"ImagePositionPatient"): rec.append((p,ds))
        except Exception: pass
    if not rec: raise RuntimeError("No Series 6 pixel DICOMs")
    first=rec[0][1]
    if int(first.SeriesNumber)!=6 or "bestsyst" not in str(first.SeriesDescription).lower():
        raise RuntimeError(f"Wrong Series 6 identity: {getattr(first,'SeriesDescription','')}")
    iop=np.asarray(first.ImageOrientationPatient,float); row,col=iop[:3],iop[3:]; normal=np.cross(row,col)
    zrec=[]
    for p,ds in rec:
        pos=np.asarray(ds.ImagePositionPatient,float); zrec.append((float(pos@normal),ds,pos))
    zrec.sort(key=lambda x:x[0])
    dedup=[]; seen=set()
    for x in zrec:
        k=round(x[0],4)
        if k not in seen: seen.add(k); dedup.append(x)
    if len(dedup)!=524: raise RuntimeError(f"Expected 524 Series 6 slices, got {len(dedup)}")
    proj=np.asarray([x[0] for x in dedup]); dz=np.abs(np.diff(proj)); dz=dz[dz>1e-4]
    ps=np.asarray(first.PixelSpacing,float); sy,sx=float(ps[0]),float(ps[1]); sz=float(np.median(dz))
    if max(sx,sy)>0.40 or sz>0.40: raise RuntimeError(f"Unexpected Series 6 spacing {(sx,sy,sz)}")
    vol=[]
    for _,ds,_ in dedup:
        try: a=ds.pixel_array.astype(np.float32)
        except Exception as e: raise RuntimeError("Install pylibjpeg and pylibjpeg-libjpeg: "+str(e)) from e
        vol.append(a*_safe_float(getattr(ds,"RescaleSlope",1),1)+_safe_float(getattr(ds,"RescaleIntercept",0),0))
    arr=np.stack(vol).astype(np.float32)
    D=np.column_stack([row,col,normal]); origin=np.asarray(dedup[0][2],float)
    g=Geometry(np.asarray([sx,sy,sz]),origin,D)
    img=sitk.GetImageFromArray(arr); img.SetSpacing(tuple(g.spacing_xyz)); img.SetOrigin(tuple(origin)); img.SetDirection(tuple(D.ravel()))
    return img,arr,g,{"series_number":6,"series_description":str(first.SeriesDescription),"n_slices":524,"spacing_xyz_mm":[sx,sy,sz]}


def _crop(ref,center,half=32.):
    idx=np.asarray(ref.TransformPhysicalPointToContinuousIndex(tuple(center)),float); sp=np.asarray(ref.GetSpacing())
    size=np.maximum(16,np.ceil(2*half/sp).astype(int)); start=np.floor(idx-size/2).astype(int); full=np.asarray(ref.GetSize())
    start=np.maximum(start,0); size=np.minimum(size,full-start)
    return sitk.RegionOfInterest(ref,[int(v) for v in size],[int(v) for v in start])


def register_root(ref,mov,center):
    f=sitk.Cast(sitk.Clamp(_crop(ref,center),lowerBound=-200,upperBound=1000),sitk.sitkFloat32)
    m=sitk.Cast(sitk.Clamp(mov,lowerBound=-200,upperBound=1000),sitk.sitkFloat32)
    r=sitk.ImageRegistrationMethod(); r.SetMetricAsMattesMutualInformation(40); r.SetMetricSamplingStrategy(r.RANDOM); r.SetMetricSamplingPercentage(.12,seed=17)
    r.SetInterpolator(sitk.sitkLinear); r.SetOptimizerAsRegularStepGradientDescent(1.0,.05,60,1e-5); r.SetOptimizerScalesFromPhysicalShift()
    r.SetShrinkFactorsPerLevel([4,2,1]); r.SetSmoothingSigmasPerLevel([2,1,0]); r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    r.SetInitialTransform(sitk.TranslationTransform(3),inPlace=False); tx=r.Execute(f,m)
    p=np.asarray(tx.GetParameters(),float); mag=float(np.linalg.norm(p[:3]))
    return tx,{"registration_success":True,"translation_x_mm":float(p[0]),"translation_y_mm":float(p[1]),"translation_z_mm":float(p[2]),"translation_magnitude_mm":mag,"metric_value":float(r.GetMetricValue()),"registration_pass":mag<=MAX_ROOT_TRANSLATION_MM}


def _native_mask(path,ref):
    img=sitk.ReadImage(str(_req(path)))
    if not _same_geometry(img,ref):
        img=sitk.Resample(img,ref,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
    return sitk.GetArrayFromImage(img)>0


def _transform_path(path,tx):
    return np.asarray([tx.TransformPoint(tuple(p)) for p in _resample_path(path,.5)],float)


def _path_component_association(labels,cid,g,path_xyz,radius=ASSOCIATION_RADIUS_MM):
    vox=np.argwhere(labels==cid)
    if not len(vox): return np.empty((0,3),int),{}
    xyz=g.zyx_to_xyz(vox)
    ptree=cKDTree(path_xyz)
    d,_=ptree.query(xyz,k=1)
    keep=d<=radius
    assoc=vox[keep]
    return assoc,{
        "component_voxels":int(len(vox)),
        "associated_voxels":int(keep.sum()),
        "associated_fraction":float(keep.mean()),
        "min_path_distance_mm":float(d.min()),
        "median_path_distance_mm":float(np.median(d)),
    }


def _nearest_gap(assoc_zyx,dt,nearest_idx,g):
    if len(assoc_zyx)==0: return None
    vals=dt[tuple(assoc_zyx.T)]
    i=int(np.argmin(vals)); z=assoc_zyx[i]; az=nearest_idx[:,z[0],z[1],z[2]]
    p=g.zyx_to_xyz([z])[0]; a=g.zyx_to_xyz([az])[0]
    return {
        "gap_distance_mm":float(vals[i]),
        "model_zyx":[int(x) for x in z],
        "aorta_zyx":[int(x) for x in az],
        "model_xyz_mm":[float(x) for x in p],
        "aorta_xyz_mm":[float(x) for x in a],
    }


def _line_samples(arr,g,p0,p1,step=GAP_SAMPLE_STEP_MM):
    p0=np.asarray(p0,float); p1=np.asarray(p1,float); length=float(np.linalg.norm(p1-p0))
    n=max(2,int(np.ceil(length/max(step,1e-6)))+1)
    t=np.linspace(0,1,n); pts=p0[None,:]+(p1-p0)[None,:]*t[:,None]
    z=g.xyz_to_zyx(pts); hu=map_coordinates(arr,z.T,order=1,mode="constant",cval=np.nan)
    return pts,hu


def _gap_hu_metrics(arr,g,gap):
    if gap is None: return {}
    p0=np.asarray(gap["model_xyz_mm"]); p1=np.asarray(gap["aorta_xyz_mm"])
    pts,hu=_line_samples(arr,g,p0,p1)
    interior=hu[1:-1] if len(hu)>2 else hu
    finite=interior[np.isfinite(interior)]
    if not len(finite):
        return {"line_samples":int(len(hu)),"interior_samples":0,"interior_median_hu":np.nan,"interior_p25_hu":np.nan,"interior_fraction_ge_150hu":0.0}
    return {
        "line_samples":int(len(hu)),
        "interior_samples":int(len(finite)),
        "interior_median_hu":float(np.median(finite)),
        "interior_p25_hu":float(np.percentile(finite,25)),
        "interior_fraction_ge_150hu":float(np.mean(finite>=CONTRAST_HU_THRESHOLD)),
    }


def _orth_views(arr,g,pt,half=18.):
    zyx=np.rint(g.xyz_to_zyx([pt])[0]).astype(int)
    rz=max(4,int(round(half/g.spacing_zyx[0]))); ry=max(4,int(round(half/g.spacing_zyx[1]))); rx=max(4,int(round(half/g.spacing_zyx[2])))
    z,y,x=zyx; z=int(np.clip(z,0,arr.shape[0]-1)); y=int(np.clip(y,0,arr.shape[1]-1)); x=int(np.clip(x,0,arr.shape[2]-1))
    return (arr[z,max(0,y-ry):min(arr.shape[1],y+ry+1),max(0,x-rx):min(arr.shape[2],x+rx+1)],
            arr[max(0,z-rz):min(arr.shape[0],z+rz+1),y,max(0,x-rx):min(arr.shape[2],x+rx+1)],
            arr[max(0,z-rz):min(arr.shape[0],z+rz+1),max(0,y-ry):min(arr.shape[1],y+ry+1),x])


def _plot_gap(ct,cor,aor,g,gap,out,title):
    p0=np.asarray(gap["model_xyz_mm"]); p1=np.asarray(gap["aorta_xyz_mm"]); mid=(p0+p1)/2
    cv=_orth_views(ct,g,mid); pv=_orth_views(cor.astype(float),g,mid); av=_orth_views(aor.astype(float),g,mid)
    fig,ax=plt.subplots(1,3,figsize=(12,4))
    for i,n in enumerate(("axial","coronal","sagittal")):
        ax[i].imshow(cv[i],cmap="gray",vmin=-100,vmax=900)
        try: ax[i].contour(av[i],levels=[.5],linewidths=1); ax[i].contour(pv[i],levels=[.5],linewidths=1)
        except Exception: pass
        ax[i].set_title(f"{title} | {n}"); ax[i].axis("off")
    fig.tight_layout(); fig.savefig(out,dpi=170); plt.close(fig)

    pts,hu=_line_samples(ct,g,p0,p1)
    fig,ax=plt.subplots(figsize=(7,4))
    d=np.linalg.norm(pts-p0,axis=1)
    ax.plot(d,hu,marker="o",markersize=2)
    ax.axhline(CONTRAST_HU_THRESHOLD,linestyle="--",linewidth=1)
    ax.set_xlabel("Distance from model endpoint toward aorta (mm)"); ax.set_ylabel("HU")
    ax.set_title(title+" | straight-gap HU profile")
    fig.tight_layout(); fig.savefig(str(out).replace("_planes.png","_hu_profile.png"),dpi=170); plt.close(fig)


def _series7_endpoint_aorta(path,g,shape,aorta):
    dt=ndi.distance_transform_edt(~aorta,sampling=g.spacing_zyx)
    pts=np.asarray([path[0],path[-1]],float); z=np.rint(g.xyz_to_zyx(pts)).astype(int)
    vals=[float(dt[tuple(q)]) if np.all((q>=0)&(q<np.asarray(shape))) else np.inf for q in z]
    return pts[int(np.argmin(vals))]


def prepare(drive_root="/content/drive/MyDrive/OpenPlaque",dicom_root="/content/drive/MyDrive/CCTA/DICOM/3221",local_workdir="/content/openplaque_series6_gap",output_dir=None):
    root=Path(drive_root); out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME; out.mkdir(parents=True,exist_ok=True)
    local=Path(local_workdir); local.mkdir(parents=True,exist_ok=True)
    _write_json(out/"run_state.json",{"status":"PREPARING","algorithm":ALGORITHM,"baseline":BASELINE})
    if _read_json(root/MASTER).get("status")!=EXPECTED_MASTER_STATUS: raise RuntimeError("Frozen master prerequisite failed")
    ref,ct7,g7=_source_reference(root); a7=_source_aorta(root,ref)
    paths={n:_load_path(root/p) for n,p in {"LAD":LAD_PATH,"RCA":RCA_PATH,"C6":C6_PATH,"C7":C7_PATH}.items()}
    rca7=_series7_endpoint_aorta(paths["RCA"],g7,ct7.shape,a7); lad7=_series7_endpoint_aorta(paths["LAD"],g7,ct7.shape,a7); center=(rca7+lad7)/2
    s6img,ct6,g6,meta=load_series6(_req(Path(dicom_root)/SERIES6_FOLDER_NAME))
    tx,reg=register_root(ref,s6img,center)
    prep={"algorithm":ALGORITHM,"baseline_commit":BASELINE,"series6_metadata":meta,"root_registration":reg,
          "root_transform_parameters":[float(x) for x in tx.GetParameters()],
          "cached_series6_prediction":str(root/"Series6_Left_Coronary_Origin_Validation_v1/external_model_predictions/series6_alternate.nii.gz"),
          "cached_series7_prediction":str(root/"Series6_Left_Coronary_Origin_Validation_v1/external_model_predictions/series7_reference.nii.gz"),
          "cached_series6_aorta":str(root/"Series6_Native_Ostium_Topology_v1/native_aorta/series6_aorta.nii.gz")}
    for k in ("cached_series6_prediction","cached_series7_prediction","cached_series6_aorta"): _req(prep[k])
    _write_json(out/"preparation.json",prep)
    _write_json(out/"input_provenance.json",prep)
    _write_json(out/"run_state.json",{"status":"PREPARED","algorithm":ALGORITHM,"root_registration_pass":bool(reg["registration_pass"])})
    return prep


def analyze(drive_root="/content/drive/MyDrive/OpenPlaque",dicom_root="/content/drive/MyDrive/CCTA/DICOM/3221",output_dir=None):
    root=Path(drive_root); out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME; prep=_read_json(out/"preparation.json")
    ref,ct7,g7=_source_reference(root); a7=_source_aorta(root,ref)
    s6img,ct6,g6,_=load_series6(_req(Path(dicom_root)/SERIES6_FOLDER_NAME))
    p6=_native_mask(prep["cached_series6_prediction"],s6img); a6=_native_mask(prep["cached_series6_aorta"],s6img)
    p7=_native_mask(prep["cached_series7_prediction"],ref)
    labels6,n6=ndi.label(p6,structure=np.ones((3,3,3),np.uint8)); labels7,n7=ndi.label(p7,structure=np.ones((3,3,3),np.uint8))
    dt6,idx6=ndi.distance_transform_edt(~a6,sampling=g6.spacing_zyx,return_indices=True)
    dt7,idx7=ndi.distance_transform_edt(~a7,sampling=g7.spacing_zyx,return_indices=True)

    paths7={n:_load_path(root/p) for n,p in {"LAD":LAD_PATH,"RCA":RCA_PATH,"C6":C6_PATH,"C7":C7_PATH}.items()}
    tx=sitk.TranslationTransform(3); tx.SetParameters(tuple(prep["root_transform_parameters"]))
    paths6={n:_transform_path(p,tx) for n,p in paths7.items()}

    rows=[]
    for series,labels,ncomp,g,paths,dt,idx,ct,cor,aor in [
        ("series6",labels6,n6,g6,paths6,dt6,idx6,ct6,p6,a6),
        ("series7",labels7,n7,g7,{n:_resample_path(p,.5) for n,p in paths7.items()},dt7,idx7,ct7,p7,a7),
    ]:
        for cid in range(1,int(ncomp)+1):
            for pname,path in paths.items():
                assoc,meta=_path_component_association(labels,cid,g,path)
                if meta["associated_voxels"]<MIN_ASSOCIATED_VOXELS: continue
                gap=_nearest_gap(assoc,dt,idx,g); hu=_gap_hu_metrics(ct,g,gap)
                row={"series":series,"component":cid,"path":pname,**meta,**gap,**hu}
                row["short_gap"]=bool(row["gap_distance_mm"]<=SHORT_GAP_MM)
                row["contrast_supported"]=bool(
                    row["short_gap"] and row.get("interior_samples",0)>0
                    and row.get("interior_fraction_ge_150hu",0)>=CONTRAST_FRACTION_THRESHOLD
                )
                rows.append(row)
                if series=="series6" and pname in ("LAD","C6","C7"):
                    _plot_gap(ct,cor,aor,g,gap,out/f"QC_series6_comp{cid}_{pname}_gap_planes.png",f"Series 6 comp {cid} {pname} gap {row['gap_distance_mm']:.2f} mm")

    df=pd.DataFrame(rows)
    df.to_csv(out/"path_associated_component_gaps.csv",index=False)
    if df.empty: raise RuntimeError("No path-associated coronary component subsets found")

    s6left=df[(df.series=="series6") & df.path.isin(["LAD","C6","C7"])].copy()
    s6rca=df[(df.series=="series6") & (df.path=="RCA")].copy()
    if s6left.empty:
        status=STATUS_LOCALIZATION_FAIL; best=None
    else:
        best=s6left.sort_values(["gap_distance_mm","associated_voxels"],ascending=[True,False]).iloc[0]
        if float(best.gap_distance_mm)<=SHORT_GAP_MM:
            status=STATUS_SHORT_CONTRAST if bool(best.contrast_supported) else STATUS_SHORT_NO_CONTRAST
        else:
            status=STATUS_NO_SHORT
    if not bool(prep["root_registration"]["registration_pass"]): status=STATUS_REG_FAIL

    best_json=None if best is None else {k:(v.item() if hasattr(v,"item") else v) for k,v in best.to_dict().items()}
    decision={"status":status,"best_series6_left_gap":best_json,
              "series6_left_short_gap":False if best is None else bool(best.gap_distance_mm<=SHORT_GAP_MM),
              "series6_left_contrast_supported":False if best is None else bool(best.contrast_supported),
              "clinical_LM_identity_established":False,"clinical_LCX_OM_identity_established":False,
              "master_anatomy_modified":False,
              "reopen_left_coronary_review":bool(status in (STATUS_SHORT_CONTRAST,STATUS_SHORT_NO_CONTRAST))}
    summary={"status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE,
             "root_registration":prep["root_registration"],
             "series6_components":int(n6),"series7_components":int(n7),
             "prespecified_gates":{"ASSOCIATION_RADIUS_MM":ASSOCIATION_RADIUS_MM,"MIN_ASSOCIATED_VOXELS":MIN_ASSOCIATED_VOXELS,
                "SHORT_GAP_MM":SHORT_GAP_MM,"GAP_SAMPLE_STEP_MM":GAP_SAMPLE_STEP_MM,
                "CONTRAST_HU_THRESHOLD":CONTRAST_HU_THRESHOLD,"CONTRAST_FRACTION_THRESHOLD":CONTRAST_FRACTION_THRESHOLD},
             "decision":decision,
             "scientific_boundary":"Diagnostic gap analysis only. No segmentation thresholds are relaxed and no synthetic bridge is created. Path association is descriptive localization; primary reported gap is the native distance from path-associated CAS-Net voxels to the native aorta."}
    _write_json(out/"decision.json",decision); _write_json(out/"summary.json",summary)

    fig,ax=plt.subplots(figsize=(9,5))
    plotdf=s6left.sort_values("gap_distance_mm")
    if not plotdf.empty:
        labels=[f"c{int(r.component)}-{r.path}" for _,r in plotdf.iterrows()]
        ax.bar(labels,plotdf.gap_distance_mm)
        ax.axhline(SHORT_GAP_MM,linestyle="--",linewidth=1)
        ax.tick_params(axis="x",rotation=60)
    ax.set_ylabel("Minimum native aorta gap (mm)"); ax.set_title("Series 6 left-associated component gaps"); fig.tight_layout(); fig.savefig(out/"01_series6_left_gap_summary.png",dpi=170); plt.close(fig)

    report=out/"OPENPLAQUE_SERIES6_NATIVE_ALL_COMPONENT_GAP_ANALYSIS_V1_REPORT.html"
    report.write_text("<html><body><h1>OpenPlaque Series 6 Native All-Component Gap Analysis v1</h1>"+f"<p><b>Status:</b> {status}</p><p>No artificial bridge is created.</p>"+df.to_html(index=False)+"<h2>Series 6 left gaps</h2><img src='01_series6_left_gap_summary.png' width='900'>"+ "".join(f"<h3>{p.stem}</h3><img src='{p.name}' width='1000'>" for p in sorted(out.glob("QC_series6_*_gap_planes.png")))+"</body></html>")
    _write_json(out/"run_state.json",{"status":"COMPLETE","algorithm":ALGORITHM,"result_status":status})
    archive=out/"OPENPLAQUE_SERIES6_NATIVE_ALL_COMPONENT_GAP_ANALYSIS_V1_RESULTS.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p!=archive: z.write(p,p.name)
    return summary


def synthetic_self_test():
    arr=np.linspace(0,300,11).reshape(11,1,1).astype(float)
    g=Geometry(np.array([1.,1.,1.]),np.zeros(3),np.eye(3))
    pts,hu=_line_samples(arr,g,np.array([0.,0.,0.]),np.array([0.,0.,10.]),1.0)
    assert len(hu)==11
    assert np.isclose(hu[-1],300)
    return {"ok":True}
