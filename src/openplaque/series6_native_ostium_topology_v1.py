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
ALGORITHM = "series6-native-ostium-topology-v1.0"
OUTPUT_DIRNAME = "Series6_Native_Ostium_Topology_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
LAD_PATH = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
RCA_PATH = Path("Cache/Source_Volume_Coronary_Centerlines/RCA_source_centerline.csv")
C6_PATH = Path("Joint_Three_Vessel_Template_Classifier_v1/candidate_04_source_path.csv")
C7_PATH = Path("LCX_Distal_Reacquisition_v1_fixed/C7_extended_path.csv")
SOURCE_AORTA = Path("TotalSegmentator_Cardiovascular_Cache_v1/heartchambers_highres/aorta.nii.gz")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
EXPECTED_MASTER_STATUS = "CORONARY_ANATOMY_BASELINE_V2_FROZEN"

SERIES6_FOLDER_NAME = "3228"
AORTA_CONTACT_TOLERANCE_MM = 1.5
MIN_CONTACT_VOXELS = 10
RCA_CONTACT_RADIUS_MM = 5.0
MIN_SECOND_CONTACT_SEPARATION_MM = 8.0
MAX_ROOT_TRANSLATION_MM = 6.0
MIN_AORTA_VOLUME_MM3 = 10_000.0
MAX_AORTA_VOLUME_MM3 = 500_000.0

STATUS_SECOND = "SERIES6_NATIVE_SECOND_AORTIC_CONTACT_PRESENT"
STATUS_SINGLE = "SERIES6_NATIVE_SINGLE_CORONARY_AORTIC_CONTACT"
STATUS_RCA_FAIL = "SERIES6_NATIVE_RCA_CONTROL_FAILED"
STATUS_AORTA_FAIL = "SERIES6_NATIVE_AORTA_SEGMENTATION_FAILED"
STATUS_REG_FAIL = "SERIES6_NATIVE_ROOT_REGISTRATION_FAILED"


def _req(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _read_json(p):
    return json.loads(_req(p).read_text())


def _write_json(p, x):
    Path(p).write_text(json.dumps(x, indent=2, default=str, allow_nan=True))


def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return float(default)


def _load_path(p):
    d = pd.read_csv(_req(p))
    for cols in (("lps_x_mm","lps_y_mm","lps_z_mm"),("x_mm","y_mm","z_mm")):
        if all(c in d.columns for c in cols):
            return d[list(cols)].to_numpy(float)
    raise ValueError(f"No supported LPS columns in {p}")


def _arc(p):
    p=np.asarray(p,float)
    return np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))] if len(p)>1 else np.zeros(len(p))


def _resample_path(p, step=.5):
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

    def xyz_to_zyx(self, pts):
        pts=np.atleast_2d(np.asarray(pts,float))
        xyz=((pts-self.origin) @ np.linalg.inv(self.direction).T)/self.spacing_xyz
        return xyz[:,::-1]

    def zyx_to_xyz(self, zyx):
        zyx=np.atleast_2d(np.asarray(zyx,float))
        return self.origin + (zyx[:,::-1]*self.spacing_xyz) @ self.direction.T


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
    return img,g,arr.shape


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


def _write_local_image(img,path):
    """Write a large transient image with explicit uncompressed SimpleITK IO.

    MetaImage (.mha) is used because it has already proved reliable for the
    full 524x512x512 CCTA volume in Colab, unlike large NIfTI writes.
    """
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    w=sitk.ImageFileWriter(); w.SetFileName(str(path)); w.SetUseCompression(False); w.Execute(img)
    if not path.exists() or path.stat().st_size==0: raise RuntimeError(f"Failed writing {path}")
    chk=sitk.ReadImage(str(path))
    if tuple(chk.GetSize())!=tuple(img.GetSize()): raise RuntimeError("Local image readback size mismatch")
    if not np.allclose(chk.GetSpacing(),img.GetSpacing(),atol=1e-5): raise RuntimeError("Local image readback spacing mismatch")
    if not np.allclose(chk.GetOrigin(),img.GetOrigin(),atol=1e-4): raise RuntimeError("Local image readback origin mismatch")
    if not np.allclose(chk.GetDirection(),img.GetDirection(),atol=1e-5): raise RuntimeError("Local image readback direction mismatch")
    return path


def _sample(g,arr,pts):
    return map_coordinates(arr,g.xyz_to_zyx(pts).T,order=1,mode="constant",cval=np.nan)


def _brightness(g,arr,paths):
    vals=[]; cov=[]
    shp=np.asarray(arr.shape)
    for p in paths:
        q=_resample_path(p,2.0); z=g.xyz_to_zyx(q); ok=np.all((z>=0)&(z<shp),axis=1); cov.append(ok.mean())
        if ok.any(): vals += _sample(g,arr,q[ok])[np.isfinite(_sample(g,arr,q[ok]))].tolist()
    return {"path_coverage_fraction":float(np.mean(cov)),"path_median_hu":float(np.median(vals)),"path_p10_hu":float(np.percentile(vals,10))}


def _source_aorta(root,ref):
    a=sitk.ReadImage(str(_req(Path(root)/SOURCE_AORTA)))
    if not _same_geometry(a,ref): a=sitk.Resample(a,ref,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
    return sitk.GetArrayFromImage(a)>0


def _endpoint_near_aorta(path,g,shape,aorta):
    dt=ndi.distance_transform_edt(~aorta,sampling=g.spacing_zyx)
    pts=np.asarray([path[0],path[-1]],float); z=np.rint(g.xyz_to_zyx(pts)).astype(int)
    ds=[float(dt[tuple(q)]) if np.all((q>=0)&(q<np.asarray(shape))) else np.inf for q in z]
    i=int(np.argmin(ds)); return pts[i],ds[i]


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


def prepare(drive_root="/content/drive/MyDrive/OpenPlaque",dicom_root="/content/drive/MyDrive/CCTA/DICOM/3221",local_workdir="/content/openplaque_series6_native",output_dir=None):
    root=Path(drive_root); out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME; out.mkdir(parents=True,exist_ok=True)
    local=Path(local_workdir); local.mkdir(parents=True,exist_ok=True)
    _write_json(out/"run_state.json",{"status":"PREPARING","algorithm":ALGORITHM,"baseline":BASELINE})
    if _read_json(root/MASTER).get("status")!=EXPECTED_MASTER_STATUS: raise RuntimeError("Frozen master prerequisite failed")
    ref,g7,shape7=_source_reference(root); a7=_source_aorta(root,ref)
    lad=_load_path(root/LAD_PATH); rca=_load_path(root/RCA_PATH); c6=_load_path(root/C6_PATH); c7=_load_path(root/C7_PATH)
    rca7,rd=_endpoint_near_aorta(rca,g7,shape7,a7); left7,ld=_endpoint_near_aorta(lad,g7,shape7,a7); center=(rca7+left7)/2
    s6img,s6arr,g6,meta=load_series6(_req(Path(dicom_root)/SERIES6_FOLDER_NAME))
    bright=_brightness(g6,s6arr,[lad,rca])
    if bright["path_coverage_fraction"]<.95 or bright["path_median_hu"]<150: raise RuntimeError(f"Series 6 source validation failed {bright}")
    tx,reg=register_root(ref,s6img,center); tf=local/"series7_to_series6_root_translation.tfm"; sitk.WriteTransform(tx,str(tf))
    local_img=_write_local_image(s6img,local/"series6_native.img.mha")
    prep={"algorithm":ALGORITHM,"baseline_commit":BASELINE,"series6_native_image":str(local_img),"series6_dicom_folder":str(Path(dicom_root)/SERIES6_FOLDER_NAME),"series6_metadata":meta,"series6_brightness":bright,
          "root_registration":reg,"root_transform_file":str(tf),"series7_rca_root_lps_mm":rca7.tolist(),"series7_left_anchor_lps_mm":left7.tolist(),
          "series6_expected_rca_root_lps_mm":list(tx.TransformPoint(tuple(rca7))),"series6_expected_left_anchor_lps_mm":list(tx.TransformPoint(tuple(left7))),
          "source_endpoint_aorta_distance_mm":{"RCA":rd,"LAD":ld}}
    _write_json(out/"preparation.json",prep)
    _write_json(out/"input_provenance.json",{"series6_folder":str(Path(dicom_root)/SERIES6_FOLDER_NAME),
        "cached_series6_prediction":str(root/"Series6_Left_Coronary_Origin_Validation_v1/external_model_predictions/series6_alternate.nii.gz"),
        "series7_source_aorta":str(root/SOURCE_AORTA),"LAD":str(root/LAD_PATH),"RCA":str(root/RCA_PATH),"C6":str(root/C6_PATH),"C7":str(root/C7_PATH)})
    _write_json(out/"run_state.json",{"status":"PREPARED","algorithm":ALGORITHM,"root_registration_pass":bool(reg["registration_pass"])})
    return prep


def _native_mask(path,native):
    img=sitk.ReadImage(str(_req(path)))
    if not _same_geometry(img,native): img=sitk.Resample(img,native,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
    return sitk.GetArrayFromImage(img)>0


def _transform_path(path,tx):
    return np.asarray([tx.TransformPoint(tuple(p)) for p in _resample_path(path,.5)],float)


def _component_assoc(labels,cid,g,paths):
    z=np.argwhere(labels==cid)
    if not len(z): return {}
    tree=cKDTree(g.zyx_to_xyz(z)); out={}
    for name,p in paths.items():
        d,_=tree.query(p,k=1)
        out[name]={"min_distance_mm":float(d.min()),"median_distance_mm":float(np.median(d)),"fraction_within_2mm":float(np.mean(d<=2)),"fraction_within_4mm":float(np.mean(d<=4))}
    return out


def _orth(arr,g,pt,half=22.):
    zyx=np.rint(g.xyz_to_zyx([pt])[0]).astype(int); rz=max(4,int(round(half/g.spacing_zyx[0]))); ry=max(4,int(round(half/g.spacing_zyx[1]))); rx=max(4,int(round(half/g.spacing_zyx[2])))
    z,y,x=zyx; z=int(np.clip(z,0,arr.shape[0]-1)); y=int(np.clip(y,0,arr.shape[1]-1)); x=int(np.clip(x,0,arr.shape[2]-1))
    return (arr[z,max(0,y-ry):min(arr.shape[1],y+ry+1),max(0,x-rx):min(arr.shape[2],x+rx+1)],
            arr[max(0,z-rz):min(arr.shape[0],z+rz+1),y,max(0,x-rx):min(arr.shape[2],x+rx+1)],
            arr[max(0,z-rz):min(arr.shape[0],z+rz+1),max(0,y-ry):min(arr.shape[1],y+ry+1),x])


def _plot_qc(ct,cor,aor,g,pt,out,title):
    fig,ax=plt.subplots(1,3,figsize=(12,4)); cv=_orth(ct,g,pt); pv=_orth(cor.astype(float),g,pt); av=_orth(aor.astype(float),g,pt)
    for i,n in enumerate(("axial","coronal","sagittal")):
        ax[i].imshow(cv[i],cmap="gray",vmin=-100,vmax=900)
        try: ax[i].contour(av[i],levels=[.5],linewidths=1); ax[i].contour(pv[i],levels=[.5],linewidths=1)
        except Exception: pass
        ax[i].set_title(f"{title} | {n}"); ax[i].axis("off")
    fig.tight_layout(); fig.savefig(out,dpi=170); plt.close(fig)


def analyze(native_aorta_path,cached_prediction_path,drive_root="/content/drive/MyDrive/OpenPlaque",local_workdir="/content/openplaque_series6_native",output_dir=None):
    root=Path(drive_root); out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME; prep=_read_json(out/"preparation.json")
    native=sitk.ReadImage(prep["series6_native_image"]); ct=sitk.GetArrayFromImage(native).astype(np.float32)
    g=Geometry(np.asarray(native.GetSpacing()),np.asarray(native.GetOrigin()),np.asarray(native.GetDirection()).reshape(3,3))
    cor=_native_mask(cached_prediction_path,native); aor=_native_mask(native_aorta_path,native)
    vv=float(np.prod(g.spacing_xyz)); avol=float(aor.sum()*vv); aplaus=MIN_AORTA_VOLUME_MM3<=avol<=MAX_AORTA_VOLUME_MM3
    labels,ncomp=ndi.label(cor,structure=np.ones((3,3,3),np.uint8))
    dt=ndi.distance_transform_edt(~aor,sampling=g.spacing_zyx); contact=cor&(dt<=AORTA_CONTACT_TOLERANCE_MM); clab,ncontact=ndi.label(contact,structure=np.ones((3,3,3),np.uint8))
    tx=sitk.ReadTransform(prep["root_transform_file"]); inv=tx.GetInverse(); rca6=np.asarray(prep["series6_expected_rca_root_lps_mm"],float); left6=np.asarray(prep["series6_expected_left_anchor_lps_mm"],float)
    frozen={n:_load_path(root/p) for n,p in {"LAD":LAD_PATH,"RCA":RCA_PATH,"C6":C6_PATH,"C7":C7_PATH}.items()}; tpaths={n:_transform_path(p,tx) for n,p in frozen.items()}
    rows=[]
    for cid in range(1,int(ncontact)+1):
        z=np.argwhere(clab==cid)
        if not len(z): continue
        xyz=g.zyx_to_xyz(z); cen=xyz.mean(axis=0); comps=labels[tuple(z.T)]; comps=comps[comps>0]; comp=int(np.bincount(comps).argmax()) if len(comps) else 0; mapped=np.asarray(inv.TransformPoint(tuple(cen)),float)
        rows.append({"contact_cluster":cid,"contact_voxels":int(len(z)),"component":comp,"component_voxels":int(np.sum(labels==comp)) if comp else 0,
                     "centroid_native_x_mm":float(cen[0]),"centroid_native_y_mm":float(cen[1]),"centroid_native_z_mm":float(cen[2]),
                     "centroid_series7_x_mm":float(mapped[0]),"centroid_series7_y_mm":float(mapped[1]),"centroid_series7_z_mm":float(mapped[2]),
                     "distance_to_native_RCA_root_mm":float(np.linalg.norm(cen-rca6)),"distance_to_native_left_anchor_mm":float(np.linalg.norm(cen-left6))})
    contacts=pd.DataFrame(rows); contacts.to_csv(out/"native_aortic_contact_clusters.csv",index=False)
    qual=contacts[contacts.contact_voxels>=MIN_CONTACT_VOXELS].copy() if not contacts.empty else contacts.copy()
    rca=None
    if not qual.empty:
        q=qual.sort_values("distance_to_native_RCA_root_mm"); rca=q.iloc[0] if float(q.iloc[0].distance_to_native_RCA_root_mm)<=RCA_CONTACT_RADIUS_MM else None
    second=[]
    if rca is not None:
        rc=np.array([rca.centroid_native_x_mm,rca.centroid_native_y_mm,rca.centroid_native_z_mm])
        for _,r in qual.iterrows():
            if int(r.contact_cluster)==int(rca.contact_cluster): continue
            c=np.array([r.centroid_native_x_mm,r.centroid_native_y_mm,r.centroid_native_z_mm]); sep=float(np.linalg.norm(c-rc))
            if sep>=MIN_SECOND_CONTACT_SEPARATION_MM:
                d=r.to_dict(); d["separation_from_RCA_contact_mm"]=sep; second.append(d)
    second_columns = [
        "contact_cluster","contact_voxels","component","component_voxels",
        "centroid_native_x_mm","centroid_native_y_mm","centroid_native_z_mm",
        "centroid_series7_x_mm","centroid_series7_y_mm","centroid_series7_z_mm",
        "distance_to_native_RCA_root_mm","distance_to_native_left_anchor_mm",
        "separation_from_RCA_contact_mm",
    ]
    pd.DataFrame(second, columns=second_columns).to_csv(
        out/"native_second_contact_candidates.csv", index=False
    )
    comps=sorted(set(int(x) for x in contacts.component.tolist() if int(x)>0)) if not contacts.empty else []
    arows=[]
    for cid in comps:
        b={"component":cid,"component_voxels":int(np.sum(labels==cid))}
        for n,v in _component_assoc(labels,cid,g,tpaths).items():
            for k,x in v.items(): b[f"{n}_{k}"]=x
        arows.append(b)
    pd.DataFrame(arows).to_csv(out/"native_component_local_association.csv",index=False)
    regpass=bool(prep["root_registration"]["registration_pass"]); rcapass=rca is not None; secondpass=len(second)>0
    pd.DataFrame([{"gate":"root_registration_pass","pass":regpass},{"gate":"native_aorta_volume_plausible","pass":aplaus},{"gate":"native_RCA_contact_control","pass":rcapass},{"gate":"native_second_contact_present","pass":secondpass}]).to_csv(out/"gates.csv",index=False)
    status=STATUS_REG_FAIL if not regpass else STATUS_AORTA_FAIL if not aplaus else STATUS_RCA_FAIL if not rcapass else STATUS_SECOND if secondpass else STATUS_SINGLE
    for _,r in contacts.iterrows():
        cen=np.array([r.centroid_native_x_mm,r.centroid_native_y_mm,r.centroid_native_z_mm]); _plot_qc(ct,cor,aor,g,cen,out/f"QC_contact_{int(r.contact_cluster):02d}_native_planes.png",f"contact {int(r.contact_cluster)}")
    fig,ax=plt.subplots(figsize=(8,5))
    if not contacts.empty:
        ax.scatter(contacts.distance_to_native_RCA_root_mm,contacts.contact_voxels,s=60)
        for _,r in contacts.iterrows(): ax.annotate(str(int(r.contact_cluster)),(r.distance_to_native_RCA_root_mm,r.contact_voxels))
    ax.axvline(RCA_CONTACT_RADIUS_MM,linestyle="--",linewidth=1); ax.axhline(MIN_CONTACT_VOXELS,linestyle="--",linewidth=1); ax.set_xlabel("Distance to native expected RCA root (mm)"); ax.set_ylabel("Contact voxels"); ax.set_title("Series 6 native coronary/aorta contacts"); fig.tight_layout(); fig.savefig(out/"01_native_contact_summary.png",dpi=170); plt.close(fig)
    decision={"status":status,"native_aorta_volume_mm3":avol,"root_registration_pass":regpass,"native_RCA_contact_control_pass":rcapass,
              "native_RCA_contact_cluster":None if rca is None else int(rca.contact_cluster),"native_second_contact_present":secondpass,
              "native_second_contact_clusters":[int(x["contact_cluster"]) for x in second],"clinical_LM_identity_established":False,
              "clinical_LCX_OM_identity_established":False,"master_anatomy_modified":False,"reopen_left_coronary_review":bool(status==STATUS_SECOND)}
    summary={"status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"series6":prep["series6_metadata"],"root_registration":prep["root_registration"],
             "native_aorta_volume_mm3":avol,"native_coronary_voxels":int(cor.sum()),"native_coronary_components":int(ncomp),"native_contact_clusters":int(ncontact),
             "prespecified_gates":{"AORTA_CONTACT_TOLERANCE_MM":AORTA_CONTACT_TOLERANCE_MM,"MIN_CONTACT_VOXELS":MIN_CONTACT_VOXELS,
              "RCA_CONTACT_RADIUS_MM":RCA_CONTACT_RADIUS_MM,"MIN_SECOND_CONTACT_SEPARATION_MM":MIN_SECOND_CONTACT_SEPARATION_MM,
              "MAX_ROOT_TRANSLATION_MM":MAX_ROOT_TRANSLATION_MM,"MIN_AORTA_VOLUME_MM3":MIN_AORTA_VOLUME_MM3,"MAX_AORTA_VOLUME_MM3":MAX_AORTA_VOLUME_MM3},
             "decision":decision,"scientific_boundary":"Primary result is native Series-6 coronary/aorta contact topology. Series-7 root translation is used only to label the RCA neighborhood and for descriptive local association; it does not warp the Series-6 coronary tree for the primary contact count. A second contact is a candidate ostium, not automatic clinical LM identity."}
    _write_json(out/"decision.json",decision); _write_json(out/"summary.json",summary)
    qc=sorted(out.glob("QC_contact_*_native_planes.png")); report=out/"OPENPLAQUE_SERIES6_NATIVE_OSTIUM_TOPOLOGY_V1_REPORT.html"
    report.write_text("<html><body><h1>OpenPlaque Series 6 Native Ostium Topology v1</h1>"+f"<p><b>Status:</b> {status}</p><p>Native aorta volume: {avol:.1f} mm3. Native contact clusters: {ncontact}.</p><p>Primary contact topology is evaluated entirely in Series 6 geometry. Frozen master unchanged.</p>"+contacts.to_html(index=False)+"<h2>Contact summary</h2><img src='01_native_contact_summary.png' width='800'>"+"".join(f"<h3>{p.stem}</h3><img src='{p.name}' width='1000'>" for p in qc)+"</body></html>")
    _write_json(out/"run_state.json",{"status":"COMPLETE","algorithm":ALGORITHM,"result_status":status})
    archive=out/"OPENPLAQUE_SERIES6_NATIVE_OSTIUM_TOPOLOGY_V1_RESULTS.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p!=archive: z.write(p,p.name)
    return summary


def synthetic_self_test():
    r=np.array([0.,0.,0.]); c1=np.array([1.,0.,0.]); c2=np.array([12.,0.,0.])
    assert np.linalg.norm(c1-r)<=RCA_CONTACT_RADIUS_MM
    assert np.linalg.norm(c2-c1)>=MIN_SECOND_CONTACT_SEPARATION_MM
    assert len(_resample_path(np.array([[0.,0.,0.],[2.,0.,0.]]),.5))==5
    return {"ok":True}
