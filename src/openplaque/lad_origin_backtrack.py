from __future__ import annotations

"""LAD-specific source-volume tracker and proximal-origin estimator.

Fresh experiment from main.  The goal is deliberately narrow: find a convincing
proximal LAD segment from source CCTA, using LAD anatomical direction plus true
orthogonal lumen QC, then estimate the proximal LAD origin as the first stable
coronary-sized point when the accepted path is followed toward the aortic root.

The TotalSegmentator aorta mask is used only as an anatomical constraint.  It is
not used as a coronary segmentation.  No LCX tracking and no plaque processing
are performed here.  Research use only.
"""

import base64
import gc
import json
import math
import os
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.filters import frangi
from skimage.graph import MCP_Geometric

from .study import OpenPlaqueStudy

ALGORITHM_VERSION = "lad-origin-backtrack-v1.0"
SOURCE_SERIES = 7


def _json_write(obj, path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rss_gb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024**3
    except Exception:
        return float("nan")


def _ram(label):
    print(f"{label}: RSS {_rss_gb():.2f} GB")


def arc_mm(path_zyx, spacing_zyx):
    p = np.asarray(path_zyx, float)
    if len(p) == 0:
        return np.zeros(0, float)
    if len(p) == 1:
        return np.zeros(1, float)
    d = np.diff(p, axis=0) * np.asarray(spacing_zyx, float)[None, :]
    return np.r_[0.0, np.cumsum(np.linalg.norm(d, axis=1))]


def resample_path(path_zyx, spacing_zyx, step_mm=0.5):
    p = np.asarray(path_zyx, float)
    if len(p) < 2:
        return p.copy()
    s = arc_mm(p, spacing_zyx)
    keep = np.r_[True, np.diff(s) > 1e-6]
    p, s = p[keep], s[keep]
    if len(p) < 2 or s[-1] <= step_mm:
        return p.copy()
    q = np.arange(0, s[-1] + 0.5 * step_mm, step_mm)
    q[-1] = min(q[-1], s[-1])
    return np.column_stack([np.interp(q, s, p[:, j]) for j in range(3)])


def _series7_files(root, extract_root="/content/full_dicom_lad_origin"):
    root = Path(root)
    src = root / "Full_DICOM.zip"
    if not src.exists():
        raise FileNotFoundError(src)
    local = Path("/content/Full_DICOM.zip")
    if not local.exists() or local.stat().st_size != src.stat().st_size:
        shutil.copyfile(src, local)
    study = OpenPlaqueStudy(str(local), extract_root=extract_root)
    match = [s for s in study.series if s["series_number"] == SOURCE_SERIES]
    if not match:
        raise RuntimeError("Source CCTA series 7 not found")
    reader = sitk.ImageSeriesReader()
    files = list(reader.GetGDCMSeriesFileNames(match[0]["folder"], match[0]["uid"]))
    if not files:
        raise RuntimeError("No DICOM files for series 7")
    return files


def stream_source_ct_to_memmap(root, out_path, reuse_sources=()):
    out_path = Path(out_path)
    meta_path = out_path.with_suffix(".json")
    if out_path.exists() and meta_path.exists():
        mm = np.load(out_path, mmap_mode="r")
        meta = _json_read(meta_path)
        if tuple(meta.get("shape", [])) == tuple(mm.shape):
            return mm, meta, "reused_own_cache"
    for candidate in reuse_sources:
        candidate = Path(candidate); cmeta = candidate.with_suffix(".json")
        if candidate.exists() and cmeta.exists():
            mm = np.load(candidate, mmap_mode="r"); meta = _json_read(cmeta)
            if tuple(meta.get("shape", [])) == tuple(mm.shape):
                return mm, meta, f"imported_validated_cache:{candidate}"
    files = _series7_files(root)
    ds0 = pydicom.dcmread(files[0], force=True)
    n, rows, cols = len(files), int(ds0.Rows), int(ds0.Columns)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.int16, shape=(n, rows, cols))
    positions = []
    for i, fp in enumerate(files):
        ds = pydicom.dcmread(fp, force=True)
        arr = ds.pixel_array.astype(np.float32, copy=False)
        slope = float(getattr(ds, "RescaleSlope", 1.0)); intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        mm[i] = np.rint(np.clip(arr * slope + intercept, -32768, 32767)).astype(np.int16)
        positions.append([float(x) for x in getattr(ds, "ImagePositionPatient", (0, 0, i))])
        if i % 100 == 0:
            mm.flush()
    mm.flush()
    ps = [float(x) for x in ds0.PixelSpacing]
    pos = np.asarray(positions, float)
    dz = float(np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1))) if len(pos) > 1 else float(getattr(ds0, "SliceThickness", 1.0))
    meta = {
        "shape": [n, rows, cols],
        "spacing_zyx": [dz, ps[0], ps[1]],
        "positions_lps_mm": positions,
        "image_orientation_patient": [float(x) for x in getattr(ds0, "ImageOrientationPatient", (1,0,0,0,1,0))],
    }
    _json_write(meta, meta_path)
    del mm; gc.collect()
    return np.load(out_path, mmap_mode="r"), meta, "recomputed_and_cached"


def source_zyx_to_lps(meta, zyx):
    p = np.atleast_2d(np.asarray(zyx, float))
    positions = np.asarray(meta["positions_lps_mm"], float)
    orient = np.asarray(meta["image_orientation_patient"], float)
    row_cos, col_cos = orient[:3], orient[3:]
    sp = np.asarray(meta["spacing_zyx"], float)
    if len(positions) > 1:
        slice_vec = (positions[-1] - positions[0]) / max(len(positions)-1, 1)
    else:
        slice_vec = np.cross(row_cos, col_cos) * sp[0]
    origin = positions[0]
    out = origin[None, :] + p[:,0,None] * slice_vec[None,:]
    out += p[:,2,None] * sp[2] * row_cos[None,:]
    out += p[:,1,None] * sp[1] * col_cos[None,:]
    return out


def _orth_basis(tangent_zyx_mm):
    t = np.asarray(tangent_zyx_mm, float); t /= max(np.linalg.norm(t), 1e-9)
    ref = np.array([1.,0.,0.]) if abs(t[0]) < 0.82 else np.array([0.,1.,0.])
    u = np.cross(t, ref); u /= max(np.linalg.norm(u), 1e-9)
    v = np.cross(t, u); v /= max(np.linalg.norm(v), 1e-9)
    return u, v


def orthogonal_plane(ct, point_zyx, tangent_zyx_mm, spacing_zyx, half_mm=6.0, pix_mm=0.18):
    u, v = _orth_basis(tangent_zyx_mm)
    c = np.arange(-half_mm, half_mm + 1e-9, pix_mm, dtype=np.float32)
    U, V = np.meshgrid(c, c, indexing="xy")
    sp = np.asarray(spacing_zyx, np.float32)
    pm = np.asarray(point_zyx, np.float32) * sp
    pos = pm[None,None,:] + U[...,None]*u.astype(np.float32) + V[...,None]*v.astype(np.float32)
    vox = pos / sp[None,None,:]
    out = np.empty(U.shape, np.float32)
    map_coordinates(ct, [vox[...,0], vox[...,1], vox[...,2]], output=out, order=1, mode="nearest", prefilter=False)
    return out, c


def plane_lumen_metrics(im, c, lo_hu=120., hi_hu=1100.):
    pix = float(abs(c[1]-c[0]))
    bright = (im >= lo_hu) & (im <= hi_hu)
    lab, _ = ndi.label(bright, structure=np.ones((3,3), np.uint8))
    cy = cx = len(c)//2
    chosen = int(lab[cy,cx])
    if chosen == 0:
        yy, xx = np.nonzero(bright)
        if len(yy):
            dist = np.hypot(c[yy], c[xx]); j = int(np.argmin(dist))
            if dist[j] <= 0.9:
                chosen = int(lab[yy[j],xx[j]])
    comp = lab == chosen if chosen > 0 else np.zeros_like(bright)
    area_px = int(comp.sum())
    radius = math.sqrt(area_px * pix * pix / math.pi) if area_px else float("nan")
    if area_px:
        yy, xx = np.nonzero(comp)
        ym, xm = float(np.mean(c[yy])), float(np.mean(c[xx]))
        offset = float(math.hypot(ym,xm))
        er = ndi.binary_erosion(comp, structure=np.ones((3,3), bool))
        perimeter_px = max(1, int((comp & ~er).sum()))
        perim = perimeter_px * pix
        circularity = float(np.clip(4*math.pi*(area_px*pix*pix)/max(perim*perim,1e-8),0,1.2))
    else:
        offset = circularity = float("nan")
    U, V = np.meshgrid(c,c,indexing="xy"); R = np.hypot(U,V)
    center_hu = float(np.median(im[R <= 0.7]))
    core_hu = float(np.median(im[R <= 1.0])); ring_hu = float(np.median(im[(R>=2.5)&(R<=4.0)]))
    return {"radius_mm":radius,"centroid_offset_mm":offset,"circularity":circularity,
            "center_hu":center_hu,"core_minus_ring_hu":core_hu-ring_hu}


def serial_lumen_qc(path, ct, spacing, rca_cal, n_samples=16, label="LAD"):
    p = resample_path(path, spacing, 0.45); s = arc_mm(p, spacing)
    if len(p) < 5 or s[-1] < 4:
        return pd.DataFrame(), {"label":label,"length_mm":float(s[-1]) if len(s) else 0.,"median_plane_score":0.,"plane_pass_fraction":0.}
    rr = float(rca_cal.get("median_radius_mm",1.6)); rh = float(rca_cal.get("median_center_hu",560.))
    rmax = min(3.25, 1.95*rr + 0.15)
    ss = np.linspace(min(1.5,0.08*s[-1]), max(min(1.5,0.08*s[-1]), s[-1]-1.0), n_samples)
    rows=[]
    for x in ss:
        i=int(np.argmin(abs(s-x))); i0=max(0,i-3); i1=min(len(p)-1,i+3)
        t=(p[i1]-p[i0])*np.asarray(spacing,float)
        if np.linalg.norm(t)<1e-7: continue
        im,c=orthogonal_plane(ct,p[i],t,spacing); m=plane_lumen_metrics(im,c); r=m["radius_mm"]
        radius_score=float(np.exp(-0.5*((r-min(max(rr*1.05,1.1),2.4))/max(0.60*rr,0.65))**2)) if np.isfinite(r) else 0.
        offset_score=float(np.exp(-0.5*(m["centroid_offset_mm"]/0.75)**2)) if np.isfinite(m["centroid_offset_mm"]) else 0.
        circ_score=float(np.clip(m["circularity"]/0.55,0,1)) if np.isfinite(m["circularity"]) else 0.
        hu_score=float(np.exp(-0.5*((m["center_hu"]-rh)/330.)**2))
        contrast_score=float(1/(1+np.exp(-(m["core_minus_ring_hu"]-20.)/75.)))
        score=.33*radius_score+.28*offset_score+.17*circ_score+.12*hu_score+.10*contrast_score
        passed=bool(np.isfinite(r) and 0.45*rr<=r<=rmax and np.isfinite(m["centroid_offset_mm"]) and m["centroid_offset_mm"]<=1.15 and np.isfinite(m["circularity"]) and m["circularity"]>=0.20 and 120<=m["center_hu"]<=1100)
        rows.append({"label":label,"arc_mm":float(s[i]),**m,"plane_score":float(score),"plane_pass":passed})
    df=pd.DataFrame(rows)
    return df,{"label":label,"length_mm":float(s[-1]),"median_plane_score":float(df.plane_score.median()) if len(df) else 0.,
               "plane_pass_fraction":float(df.plane_pass.mean()) if len(df) else 0.,"median_radius_mm":float(df.radius_mm.median()) if len(df) else float("nan"),
               "median_offset_mm":float(df.centroid_offset_mm.median()) if len(df) else float("nan"),"median_circularity":float(df.circularity.median()) if len(df) else float("nan"),
               "median_center_hu":float(df.center_hu.median()) if len(df) else float("nan"),"median_core_minus_ring_hu":float(df.core_minus_ring_hu.median()) if len(df) else float("nan")}


def build_lad_evidence(ct, aorta, spacing, seed_source_zyx, half_mm=(86,78,78), target_mm=1.0):
    sp=np.asarray(spacing,float); seed=np.asarray(seed_source_zyx,float)
    half=np.ceil(np.asarray(half_mm,float)/sp).astype(int); o=np.rint(seed).astype(int)
    lo=np.maximum(0,o-half); hi=np.minimum(np.asarray(ct.shape),o+half+1)
    sl=tuple(slice(int(lo[d]),int(hi[d])) for d in range(3))
    crop=np.asarray(ct[sl],dtype=np.int16); acrop=np.asarray(aorta[sl],dtype=np.uint8)
    zoom=np.minimum(1.0,sp/float(target_mm)); ds_sp=sp/zoom
    dct=ndi.zoom(crop,zoom=zoom,order=1,mode="nearest",prefilter=False).astype(np.float32)
    daorta=ndi.zoom(acrop,zoom=zoom,order=0,mode="nearest",prefilter=False)>0
    del crop, acrop; gc.collect()
    sm=gaussian_filter(dct,sigma=np.maximum(0.55/ds_sp,0.45)).astype(np.float32)
    intensity=np.clip((sm-140.)/560.,0,1); intensity[(sm<120)|(sm>1050)]=0
    norm=np.clip((sm+100.)/1050.,0,1)
    vessel=np.nan_to_num(frangi(norm,sigmas=(0.7,1.0,1.4,1.8,2.2),black_ridges=False)).astype(np.float32)
    pos=vessel[vessel>0]; v99=np.percentile(pos,99.2) if pos.size else 1.; vessel=np.clip(vessel/max(v99,1e-8),0,1).astype(np.float32)
    small=gaussian_filter(sm,sigma=np.maximum(0.8/ds_sp,0.5)); broad=gaussian_filter(sm,sigma=np.maximum(4.0/ds_sp,1.5))
    dog=np.maximum(small-broad,0); p99=np.percentile(dog[np.isfinite(dog)],99.2) if np.any(np.isfinite(dog)) else 1.; dog=np.clip(dog/max(p99,1e-6),0,1).astype(np.float32)
    support=(.46*intensity+.42*vessel+.12*dog).astype(np.float32)
    bright=sm>=150; lumen_r=ndi.distance_transform_edt(bright,sampling=ds_sp).astype(np.float32)
    dist_aorta=ndi.distance_transform_edt(~daorta,sampling=ds_sp).astype(np.float32)
    large=np.clip((lumen_r-3.2)/2.2,0,1).astype(np.float32)
    cost=(1./(.035+support)+11.*large).astype(np.float32)
    cost[sm<120]+=20; cost[sm>1100]+=12; cost[daorta]=1e5; cost[(dist_aorta<1.2)&(~daorta)]+=20
    return {"ct":dct,"aorta":daorta,"support":support,"vesselness":vessel,"dog":dog,"lumen_radius_mm":lumen_r,"dist_aorta_mm":dist_aorta,"cost":cost,
            "lo_source_zyx":lo.astype(int),"zoom_zyx":zoom.astype(float),"spacing_zyx":ds_sp.astype(float)}


def _ds_to_source(p,evd):
    return np.asarray(evd["lo_source_zyx"],float)+np.asarray(p,float)/np.asarray(evd["zoom_zyx"],float)


def _source_to_ds(p,evd):
    return (np.asarray(p,float)-np.asarray(evd["lo_source_zyx"],float))*np.asarray(evd["zoom_zyx"],float)


def _sample_pool(evd,meta,root_source,kind="distal",max_points=18):
    sup=evd["support"]; ct=evd["ct"]; lr=evd["lumen_radius_mm"]; da=evd["dist_aorta_mm"]
    m=(ct>=140)&(ct<=1000)&(lr>=0.45)&(lr<=3.5)&(da>=2.0)
    vals=sup[m];
    if vals.size==0: return []
    m &= sup>=np.percentile(vals,82 if kind=="distal" else 77)
    coords=np.argwhere(m)
    if len(coords)==0: return []
    src=_ds_to_source(coords,evd); lps=source_zyx_to_lps(meta,src); root_lps=source_zyx_to_lps(meta,np.asarray(root_source)[None,:])[0]
    d=lps-root_lps[None,:]; inferior=-d[:,2]; anterior=-d[:,1]; left=d[:,0]; rad=np.linalg.norm(d,axis=1)
    if kind=="distal":
        ok=(inferior>=15)&(inferior<=85)&(anterior>=-8)&(anterior<=58)&(np.abs(left)<=58)&(rad>=20)&(rad<=100)
        anat=np.clip((inferior-15)/55,0,1)+.45*np.clip((anterior+5)/45,0,1)
    else:
        ok=(inferior>=-12)&(inferior<=42)&(anterior>=-18)&(anterior<=48)&(np.abs(left)<=48)&(rad>=5)&(rad<=52)
        anat=np.clip((52-rad)/47,0,1)+.25*np.clip((anterior+18)/40,0,1)
    idx=np.where(ok)[0]
    if not len(idx): return []
    score=sup[tuple(coords[idx].T)]+.28*evd["vesselness"][tuple(coords[idx].T)]+.12*anat[idx]-.08*np.maximum(lr[tuple(coords[idx].T)]-2.5,0)
    order=idx[np.argsort(score)[::-1]]
    picked=[]
    for j in order:
        p=coords[j].astype(float)
        if all(np.linalg.norm((p-q)*evd["spacing_zyx"])>=7.0 for q in picked):
            picked.append(p)
            if len(picked)>=max_points: break
    return picked


def _trace_pair(evd,start,end,pad_mm=13.):
    sp=np.asarray(evd["spacing_zyx"],float); a=np.rint(start).astype(int); b=np.rint(end).astype(int)
    pad=np.ceil(pad_mm/sp).astype(int); lo=np.maximum(0,np.minimum(a,b)-pad); hi=np.minimum(np.asarray(evd["cost"].shape),np.maximum(a,b)+pad+1)
    if np.any(hi-lo<3): return None
    local=evd["cost"][lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]
    sa=tuple((a-lo).tolist()); sb=tuple((b-lo).tolist())
    try:
        mcp=MCP_Geometric(np.asarray(local,np.float32),fully_connected=True)
        mcp.find_costs([sa]); p=np.asarray(mcp.traceback(sb),int)+lo[None,:]
    except Exception:
        return None
    return p if len(p)>=2 else None


def _orient_proximal_to_distal(path_ds,evd,meta,root_source):
    src=_ds_to_source(path_ds,evd); lps=source_zyx_to_lps(meta,src); root_lps=source_zyx_to_lps(meta,np.asarray(root_source)[None,:])[0]
    d=np.linalg.norm(lps-root_lps[None,:],axis=1)
    return path_ds[::-1].copy() if d[0]>d[-1] else path_ds.copy()


def _route_metrics(path_ds,evd,meta,root_source):
    p=_orient_proximal_to_distal(path_ds,evd,meta,root_source); sp=evd["spacing_zyx"]; s=arc_mm(p,sp); L=float(s[-1])
    su=evd["support"][tuple(p.T)]; lr=evd["lumen_radius_mm"][tuple(p.T)]; da=evd["dist_aorta_mm"][tuple(p.T)]; hv=evd["ct"][tuple(p.T)]
    src=_ds_to_source(p,evd); lps=source_zyx_to_lps(meta,src); dl=np.diff(lps,axis=0)
    total=lps[-1]-lps[0]; inferior=float(-total[2]); anterior=float(-total[1]); left=float(total[0])
    inferior_step=float(np.mean(dl[:,2]<=0.8)) if len(dl) else 0.; anterior_step=float(np.mean(dl[:,1]<=0.8)) if len(dl) else 0.
    score=.32*float(np.mean(su))+.18*float(np.percentile(su,15))+.10*inferior_step+.06*anterior_step+.12*np.clip(inferior/45,0,1)+.06*np.clip((anterior+5)/35,0,1)+.08*(1-float(np.mean(lr>3.6)))+.08*(1-float(np.mean(da<1.2)))
    return p,{"length_mm":L,"mean_support":float(np.mean(su)),"p15_support":float(np.percentile(su,15)),"median_lumen_radius_ds_mm":float(np.median(lr)),"large_lumen_fraction":float(np.mean(lr>3.6)),
              "aorta_reentry_fraction":float(np.mean(da<1.2)),"mean_hu_ds":float(np.mean(hv)),"inferior_displacement_mm":inferior,"anterior_displacement_mm":anterior,"left_displacement_mm":left,
              "inferior_step_fraction":inferior_step,"anterior_step_fraction":anterior_step,"graph_anatomy_score":float(score)}


class LADOriginBacktrackWorkflow:
    COMPONENTS=("source_ct","lad_evidence","rca_calibration","candidate_routes","candidate_qc","figures","report")
    def __init__(self,root="/content/drive/MyDrive/OpenPlaque",reuse=None):
        self.root=Path(root); self.cache=self.root/"Cache"/"LAD_Origin_Backtrack_v1"; self.out=self.root/"LAD_Origin_Backtrack_Report"
        self.cache.mkdir(parents=True,exist_ok=True); self.out.mkdir(parents=True,exist_ok=True)
        self.reuse={k:True for k in self.COMPONENTS};
        if reuse: self.reuse.update({k:bool(v) for k,v in reuse.items()})
        self.prov=[]; self.ct=self.meta=self.spacing=None; self.aorta=None; self.rca=None; self.evd=None; self.rca_cal=self.rca_qc=None
        self.routes=[]; self.route_table=None; self.best=None; self.best_qc=None; self.best_summary=None; self.origin=None
    def _record(self,c,a,p="",note=""):
        self.prov.append({"component":c,"reuse_requested":self.reuse[c],"action":a,"path":str(p),"note":note}); pd.DataFrame(self.prov).to_csv(self.out/"cache_provenance.csv",index=False)
    def cache_status(self):
        names={"source_ct":"series7_int16.npy","lad_evidence":"lad_evidence.npz","rca_calibration":"rca_calibration.json","candidate_routes":"route_candidates.csv","candidate_qc":"best_lad_summary.json","figures":"figures.done","report":"report.done"}
        return pd.DataFrame([{"component":c,"reuse":self.reuse[c],"cache_exists":(self.cache/names[c]).exists()} for c in self.COMPONENTS])
    def load_source_ct(self):
        own=self.cache/"series7_int16.npy"; priors=[self.root/"Cache"/"Left_Main_Bifurcation_v1"/"series7_int16.npy",self.root/"Cache"/"Left_Coronary_Ostium_Neck_v1"/"series7_int16.npy"]
        if not self.reuse["source_ct"]:
            for fp in (own,own.with_suffix(".json")):
                if fp.exists(): fp.unlink()
            priors=[]
        self.ct,self.meta,action=stream_source_ct_to_memmap(self.root,own,reuse_sources=priors); self.spacing=np.asarray(self.meta["spacing_zyx"],float)
        self._record("source_ct",action,own if action=="reused_own_cache" or action=="recomputed_and_cached" else action.split(":",1)[-1]); _ram("After source CT")
        return self.ct
    def build_evidence(self):
        fp=self.cache/"lad_evidence.npz"; rca_fp=self.root/"PCAT_RCA_10_50"/"rca_centerline_smoothed_zyx.csv"
        if self.ct is None: self.load_source_ct()
        self.rca=pd.read_csv(rca_fp)[["z","y","x"]].to_numpy(float)
        aorta_fp=self.root/"RCA_Ostium_TotalSegmentator"/"aorta_series7_totalseg.nii.gz"
        if self.reuse["lad_evidence"] and fp.exists():
            z=np.load(fp,allow_pickle=False); self.evd={k:z[k] for k in z.files}; self._record("lad_evidence","reused",fp); return self.evd
        aimg=sitk.ReadImage(str(aorta_fp)); self.aorta=sitk.GetArrayFromImage(aimg).astype(bool)
        if self.aorta.shape!=self.ct.shape: raise RuntimeError(f"Aorta mask shape {self.aorta.shape} != CT {self.ct.shape}")
        self.evd=build_lad_evidence(self.ct,self.aorta,self.spacing,self.rca[0]); np.savez_compressed(fp,**self.evd)
        self._record("lad_evidence","recomputed_and_cached",fp,"TotalSegmentator aorta used only as constraint"); self.aorta=None; gc.collect(); _ram("After LAD evidence")
        return self.evd
    def calibrate_rca(self):
        fpj=self.cache/"rca_calibration.json"; fpc=self.cache/"rca_serial_qc.csv"
        if self.rca is None: self.build_evidence()
        if self.reuse["rca_calibration"] and fpj.exists() and fpc.exists():
            self.rca_cal=_json_read(fpj); self.rca_qc=pd.read_csv(fpc); self._record("rca_calibration","reused",fpj); return self.rca_cal
        qdf,_=serial_lumen_qc(self.rca,self.ct,self.spacing,{"median_radius_mm":1.6,"median_center_hu":560.},n_samples=12,label="RCA_REFERENCE")
        good=qdf[np.isfinite(qdf.radius_mm)]
        self.rca_cal={"median_radius_mm":float(good.radius_mm.median()),"median_center_hu":float(good.center_hu.median()),"median_offset_mm":float(good.centroid_offset_mm.median()),"median_circularity":float(good.circularity.median()),"median_core_minus_ring_hu":float(good.core_minus_ring_hu.median()),"median_plane_score":float(good.plane_score.median()),"plane_pass_fraction":float(good.plane_pass.mean())}
        qdf.to_csv(fpc,index=False); _json_write(self.rca_cal,fpj); self.rca_qc=qdf; self._record("rca_calibration","recomputed_and_cached",fpj); return self.rca_cal
    def build_candidate_routes(self):
        table_fp=self.cache/"route_candidates.csv"; cdir=self.cache/"route_paths"
        if self.evd is None: self.build_evidence()
        if self.reuse["candidate_routes"] and table_fp.exists() and cdir.exists():
            tab=pd.read_csv(table_fp); self.routes=[]
            for i,row in tab.iterrows():
                pp=cdir/f"route_{i+1:02d}.npy"
                if pp.exists(): self.routes.append({"path_ds":np.load(pp),"rec":row.to_dict()})
            self.route_table=tab; self._record("candidate_routes","reused",table_fp); return tab
        distal=_sample_pool(self.evd,self.meta,self.rca[0],"distal",max_points=14); prox=_sample_pool(self.evd,self.meta,self.rca[0],"proximal",max_points=18)
        rows=[]; cand=[]
        for di,d in enumerate(distal):
            for pi,p in enumerate(prox):
                path=_trace_pair(self.evd,p,d,pad_mm=13.)
                if path is None: continue
                path,met=_route_metrics(path,self.evd,self.meta,self.rca[0])
                if met["length_mm"]<18 or met["length_mm"]>95: continue
                if met["inferior_displacement_mm"]<12: continue
                if met["large_lumen_fraction"]>0.28 or met["aorta_reentry_fraction"]>0.04: continue
                rec={"distal_seed_rank":di+1,"proximal_seed_rank":pi+1,**met}; rows.append(rec); cand.append({"path_ds":path,"rec":rec})
        if cand:
            order=np.argsort([c["rec"]["graph_anatomy_score"] for c in cand])[::-1][:20]; self.routes=[cand[i] for i in order]
        else: self.routes=[]
        self.route_table=pd.DataFrame([c["rec"] for c in self.routes])
        if len(self.route_table): self.route_table.insert(0,"graph_rank",np.arange(1,len(self.route_table)+1))
        self.route_table.to_csv(table_fp,index=False); shutil.rmtree(cdir,ignore_errors=True); cdir.mkdir(parents=True,exist_ok=True)
        for i,c in enumerate(self.routes,1): np.save(cdir/f"route_{i:02d}.npy",c["path_ds"].astype(np.int16))
        self._record("candidate_routes","recomputed_and_cached",table_fp,f"{len(self.routes)} LAD-directed graph routes"); _ram("After candidate routes")
        return self.route_table
    def qc_candidates(self):
        sf=self.cache/"best_lad_summary.json"; qf=self.cache/"best_lad_serial_qc.csv"; pf=self.cache/"best_lad_centerline.csv"; of=self.cache/"lad_origin_estimate.json"
        if self.reuse["candidate_qc"] and sf.exists() and qf.exists() and pf.exists() and of.exists():
            self.best_summary=_json_read(sf); self.best_qc=pd.read_csv(qf); self.best=pd.read_csv(pf)[["z","y","x"]].to_numpy(float); self.origin=_json_read(of); self._record("candidate_qc","reused",sf); return self.best_summary
        if self.route_table is None: self.build_candidate_routes()
        if self.rca_cal is None: self.calibrate_rca()
        scored=[]; rows=[]
        for i,c in enumerate(self.routes[:14],1):
            src=_ds_to_source(c["path_ds"],self.evd); src=resample_path(src,self.spacing,.5)
            qdf,qsum=serial_lumen_qc(src,self.ct,self.spacing,self.rca_cal,n_samples=16,label=f"LAD_{i}")
            anat=float(c["rec"]["graph_anatomy_score"]); combined=.52*qsum["plane_pass_fraction"]+.28*qsum["median_plane_score"]+.20*anat
            status="PASS" if (qsum["length_mm"]>=18 and qsum["plane_pass_fraction"]>=.65 and qsum["median_plane_score"]>=.55 and c["rec"]["inferior_displacement_mm"]>=12) else ("REVIEW" if qsum["plane_pass_fraction"]>=.45 and qsum["median_plane_score"]>=.48 else "FAIL")
            rec={"candidate":i,"combined_score":float(combined),"status":status,**c["rec"],**{f"serial_{k}":v for k,v in qsum.items() if k!="label"}}; rows.append(rec); scored.append((combined,status,src,qdf,qsum,c))
        pd.DataFrame(rows).sort_values("combined_score",ascending=False).to_csv(self.out/"lad_candidate_qc.csv",index=False) if rows else pd.DataFrame().to_csv(self.out/"lad_candidate_qc.csv",index=False)
        if not scored:
            self.best=None; self.best_qc=pd.DataFrame(); self.origin={"status":"NO_CANDIDATE"}; self.best_summary={"status":"NO_CANDIDATE","algorithm":ALGORITHM_VERSION,"message":"No LAD-directed route survived graph/anatomy screening."}
        else:
            scored.sort(key=lambda x:x[0],reverse=True); combined,status,src,qdf,qsum,c=scored[0]; self.best=src; self.best_qc=qdf; self.best_summary=dict(qsum); self.best_summary.update({"combined_score":float(combined),"status":status,"algorithm":ALGORITHM_VERSION,"graph_anatomy_score":float(c["rec"]["graph_anatomy_score"]),"inferior_displacement_mm":float(c["rec"]["inferior_displacement_mm"]),"anterior_displacement_mm":float(c["rec"]["anterior_displacement_mm"])})
            # Proximal LAD origin estimate = earliest point with stable nearby plane QC on root-oriented path.
            p=resample_path(self.best,self.spacing,.45); s=arc_mm(p,self.spacing); root_lps=source_zyx_to_lps(self.meta,self.rca[0][None,:])[0]; lps=source_zyx_to_lps(self.meta,p); dist=np.linalg.norm(lps-root_lps[None,:],axis=1)
            if dist[0]>dist[-1]: p=p[::-1].copy(); s=arc_mm(p,self.spacing); lps=source_zyx_to_lps(self.meta,p); dist=np.linalg.norm(lps-root_lps[None,:],axis=1)
            origin_idx=0
            # Match QC samples to path and choose earliest passing plane followed by another passing plane within ~4 mm.
            if len(qdf):
                passed=qdf[qdf.plane_pass]
                if len(passed):
                    for a in passed.arc_mm.to_numpy(float):
                        near=passed[(passed.arc_mm>=a)&(passed.arc_mm<=a+4.5)]
                        if len(near)>=2:
                            origin_idx=int(np.argmin(abs(s-a))); break
            self.best=p; self.origin={"status":"ESTIMATED" if status!="FAIL" else "UNVALIDATED","source_zyx":[float(x) for x in p[origin_idx]],"lps_mm":[float(x) for x in lps[origin_idx]],"arc_mm_from_proximal_path_start":float(s[origin_idx]),"distance_to_rca_root_reference_mm":float(dist[origin_idx]),"note":"Estimated proximal LAD takeoff point; not a manually labeled bifurcation and not an aortic ostium."}
            pd.DataFrame({"arc_mm":arc_mm(self.best,self.spacing),"z":self.best[:,0],"y":self.best[:,1],"x":self.best[:,2]}).to_csv(pf,index=False); qdf.to_csv(qf,index=False)
        _json_write(self.best_summary,sf); _json_write(self.origin,of); self._record("candidate_qc","recomputed_and_cached",sf); _ram("After LAD QC")
        return self.best_summary
    def plot_qc(self):
        done=self.cache/"figures.done"; figs=[self.out/f for f in ("01_lad_candidate_mips.png","02_top_candidate_cross_sections.png","03_rca_vs_best_lad.png","04_lad_lps_course.png","05_proximal_origin_zoom.png")]
        if self.reuse["figures"] and done.exists() and all(f.exists() for f in figs): self._record("figures","reused",done); return figs
        if self.best_summary is None: self.qc_candidates()
        # Evidence MIPs with top routes.
        fig,axs=plt.subplots(1,3,figsize=(16,5)); ims=[self.evd["ct"].max(0),self.evd["ct"].max(1),self.evd["ct"].max(2)]
        for ax,im,title in zip(axs,ims,("Axial MIP","Coronal MIP","Sagittal MIP")): ax.imshow(im,cmap="gray",vmin=-100,vmax=850,origin="lower"); ax.set_title(title); ax.axis("off")
        for rank,c in enumerate(self.routes[:8],1):
            p=c["path_ds"]; axs[0].plot(p[:,2],p[:,1],lw=2 if rank==1 else 1,alpha=.9 if rank==1 else .5); axs[1].plot(p[:,2],p[:,0],lw=2 if rank==1 else 1,alpha=.9 if rank==1 else .5); axs[2].plot(p[:,1],p[:,0],lw=2 if rank==1 else 1,alpha=.9 if rank==1 else .5)
        fig.suptitle("LAD-directed source-volume candidate routes"); fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(figs[0],dpi=180,bbox_inches="tight"); plt.close(fig)
        # Top candidate sections.
        fig,axs=plt.subplots(4,4,figsize=(11,11)); axs=axs.ravel(); top=self.routes[:4]
        for r,cand in enumerate(top):
            p=_ds_to_source(cand["path_ds"],self.evd); p=resample_path(p,self.spacing,.5); s=arc_mm(p,self.spacing); ss=np.linspace(min(2.,s[-1]*.1),max(min(2.,s[-1]*.1),s[-1]-1.),4)
            for col,x in enumerate(ss):
                ax=axs[r*4+col]; i=int(np.argmin(abs(s-x))); i0=max(0,i-3); i1=min(len(p)-1,i+3); t=(p[i1]-p[i0])*self.spacing; im,c=orthogonal_plane(self.ct,p[i],t,self.spacing); ax.imshow(im,cmap="gray",vmin=-150,vmax=850,extent=[c[0],c[-1],c[0],c[-1]],origin="lower"); ax.plot(0,0,"+",ms=8); ax.set_aspect("equal"); ax.set_title(f"#{r+1} {s[i]:.1f} mm")
        for ax in axs[len(top)*4:]: ax.axis("off")
        fig.suptitle("Top LAD candidates — true orthogonal sections"); fig.tight_layout(rect=[0,0,1,.96]); fig.savefig(figs[1],dpi=180,bbox_inches="tight"); plt.close(fig)
        # RCA vs best LAD.
        fig,axs=plt.subplots(2,8,figsize=(18,5.5))
        for r,(label,p) in enumerate((("RCA reference",self.rca),("Best LAD",self.best))):
            if p is None:
                for ax in axs[r]: ax.axis("off")
                continue
            p=resample_path(p,self.spacing,.5); s=arc_mm(p,self.spacing); ss=np.linspace(min(2.,s[-1]*.08),max(min(2.,s[-1]*.08),s[-1]-1.),8)
            for col,x in enumerate(ss):
                i=int(np.argmin(abs(s-x))); i0=max(0,i-3); i1=min(len(p)-1,i+3); t=(p[i1]-p[i0])*self.spacing; im,c=orthogonal_plane(self.ct,p[i],t,self.spacing); ax=axs[r,col]; ax.imshow(im,cmap="gray",vmin=-150,vmax=850,extent=[c[0],c[-1],c[0],c[-1]],origin="lower"); ax.plot(0,0,"+",ms=8); ax.set_aspect("equal"); ax.set_title(f"{label}\n{s[i]:.1f} mm")
        fig.suptitle(f"RCA reference vs LAD candidate — {self.best_summary.get('status')}"); fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(figs[2],dpi=180,bbox_inches="tight"); plt.close(fig)
        # LPS course.
        fig,axs=plt.subplots(1,3,figsize=(15,4.8)); root=source_zyx_to_lps(self.meta,self.rca[0][None,:])[0]
        if self.best is not None:
            l=source_zyx_to_lps(self.meta,self.best); pairs=[(0,2,"Left-right vs superior-inferior"),(1,2,"Posterior-anterior vs superior-inferior"),(0,1,"Left-right vs posterior-anterior")]
            for ax,(a,b,title) in zip(axs,pairs): ax.plot(l[:,a],l[:,b],".-"); ax.plot(root[a],root[b],"x",ms=10,label="RCA-root reference"); ax.set_title(title); ax.set_aspect("equal",adjustable="datalim"); ax.legend(fontsize=8)
        else:
            for ax in axs: ax.axis("off")
        fig.suptitle("Best LAD course in patient LPS coordinates"); fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(figs[3],dpi=180,bbox_inches="tight"); plt.close(fig)
        # Proximal origin zoom.
        fig,axs=plt.subplots(2,4,figsize=(11,5.5)); axs=axs.ravel()
        if self.best is not None:
            p=resample_path(self.best,self.spacing,.4); s=arc_mm(p,self.spacing); ss=np.linspace(0,min(12.,s[-1]),8)
            for ax,x in zip(axs,ss):
                i=int(np.argmin(abs(s-x))); i0=max(0,i-3); i1=min(len(p)-1,i+3); t=(p[i1]-p[i0])*self.spacing; im,c=orthogonal_plane(self.ct,p[i],t,self.spacing); ax.imshow(im,cmap="gray",vmin=-150,vmax=850,extent=[c[0],c[-1],c[0],c[-1]],origin="lower"); ax.plot(0,0,"+",ms=8); ax.set_aspect("equal"); ax.set_title(f"prox {s[i]:.1f} mm")
        else:
            for ax in axs: ax.axis("off")
        fig.suptitle("Proximal LAD / origin-estimate neighborhood"); fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(figs[4],dpi=180,bbox_inches="tight"); plt.close(fig)
        done.write_text(ALGORITHM_VERSION); self._record("figures","recomputed_and_cached",done); return figs
    def package(self):
        done=self.cache/"report.done"; zpath=self.out/"OPENPLAQUE_LAD_ORIGIN_BACKTRACK_REPORT_BACK.zip"
        if self.reuse["report"] and done.exists() and zpath.exists(): self._record("report","reused",zpath); return zpath
        figs=self.plot_qc(); html=self.out/"OPENPLAQUE_LAD_ORIGIN_BACKTRACK_REPORT.html"
        def img(fp):
            b64=base64.b64encode(Path(fp).read_bytes()).decode(); return f"<h2>{Path(fp).name}</h2><img style='max-width:100%' src='data:image/png;base64,{b64}'>"
        tab=pd.read_csv(self.out/"lad_candidate_qc.csv") if (self.out/"lad_candidate_qc.csv").exists() else pd.DataFrame()
        html.write_text("<html><body><h1>OpenPlaque — LAD origin backtracking</h1><p><b>Research use only.</b> The TotalSegmentator aorta mask is used only as an anatomical constraint. No LCX tracking and no plaque processing are performed.</p>"+"".join(img(f) for f in figs)+"<h2>Best LAD summary</h2><pre>"+json.dumps(self.best_summary,indent=2)+"</pre><h2>Estimated proximal LAD origin</h2><pre>"+json.dumps(self.origin,indent=2)+"</pre><h2>Candidate ranking</h2>"+tab.head(20).to_html(index=False)+"</body></html>",encoding="utf-8")
        names=[f.name for f in figs]+["cache_provenance.csv","lad_candidate_qc.csv",html.name]
        for fp in (self.cache/"route_candidates.csv",self.cache/"best_lad_summary.json",self.cache/"best_lad_serial_qc.csv",self.cache/"best_lad_centerline.csv",self.cache/"lad_origin_estimate.json",self.cache/"rca_calibration.json",self.cache/"rca_serial_qc.csv"):
            if fp.exists(): shutil.copyfile(fp,self.out/fp.name); names.append(fp.name)
        with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
            for n in dict.fromkeys(names):
                fp=self.out/n
                if fp.exists(): z.write(fp,arcname=n)
        done.write_text(ALGORITHM_VERSION); self._record("report","recomputed_and_cached",zpath); return zpath
