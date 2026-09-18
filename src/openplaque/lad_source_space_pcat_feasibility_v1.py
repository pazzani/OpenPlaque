from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "lad-source-space-pcat-feasibility-v1.0"
OUTPUT_DIRNAME = "LAD_Source_Space_PCAT_Feasibility_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
FROZEN_LAD = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
DISTAL_EXTENSION = Path("LAD_Distal_Endpoint_Continuation_v1/best_independent_distal_extension.csv")
DISTAL_BIDIR_SUMMARY = Path("LAD_Distal_Bidirectional_Validation_v1/summary.json")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")

EXPECTED_MASTER_STATUS = "CORONARY_ANATOMY_BASELINE_V2_FROZEN"
EXPECTED_BIDIR_STATUS = "LAD_DISTAL_EXTENSION_BIDIRECTIONALLY_SOURCE_CONFIRMED"

ARC_STEP_MM = 0.50
N_ANGLES = 72
PREVIEW_ANGLES = 36
RADIAL_STEP_MM = 0.10
PREVIEW_RADIAL_STEP_MM = 0.15
LUMEN_MAX_RADIUS_MM = 4.0
LUMEN_MIN_RADIUS_MM = 0.55
CENTER_SEARCH_RADIUS_MM = 0.75
CENTER_RING_RADII_MM = (0.25, 0.50, 0.75)
CENTER_RING_DIRECTIONS = 8
TANGENT_HALF_WINDOW_STATIONS = 3

FAT_HU_RANGE = (-190.0, -30.0)
PRIMARY_WALL_MARGIN_MM = 0.75
MARGINS_MM = (0.25, 0.50, 0.75, 1.00, 1.25)

# frozen_arc_mm is 0 at the frozen distal endpoint and increases toward the proximal frozen endpoint.
FROZEN_SEGMENT_RANGE_MM = (1.0, 24.0)
DISTAL_SEGMENT_RANGE_MM = (-14.0, -1.0)

MIN_FROZEN_SEGMENT_LENGTH_MM = 20.0
MIN_DISTAL_SEGMENT_LENGTH_MM = 10.0
MIN_FAT_VOXELS = 1000
MIN_LONGITUDINAL_COVERAGE = 0.90
MIN_BIN_FAT_VOXELS = 25

STATUS_PASS = "LAD_SOURCE_SPACE_PCAT_FEASIBILITY_COMPLETE"
STATUS_FAIL = "LAD_SOURCE_SPACE_PCAT_FEASIBILITY_FAILED"


def _req(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _read_json(p):
    return json.loads(_req(p).read_text(encoding="utf-8"))


def _write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")


def _load_path(p):
    d = pd.read_csv(_req(p))
    for cols in (("lps_x_mm","lps_y_mm","lps_z_mm"),("x_mm","y_mm","z_mm")):
        if all(c in d.columns for c in cols):
            return d[list(cols)].to_numpy(float)
    raise ValueError(f"No LPS coordinates in {p}")


def _arc(p):
    p = np.asarray(p, float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))] if len(p) > 1 else np.zeros(len(p))


def _interp(p, q):
    a = _arc(p)
    q = np.asarray(q, float)
    return np.column_stack([np.interp(q, a, p[:,k]) for k in range(3)])


def _resample(p, step=ARC_STEP_MM):
    a = _arc(p)
    q = np.arange(0.0, float(a[-1]) + 1e-9, float(step))
    if q[-1] < a[-1] - 1e-6:
        q = np.r_[q, a[-1]]
    return _interp(p, q), q


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v/n if n > 1e-9 else np.zeros_like(v)


def _orth_basis(t):
    t = _unit(t)
    axes = np.eye(3)
    seed = axes[np.argmin(np.abs(axes @ t))]
    u = _unit(np.cross(t, seed))
    v = _unit(np.cross(t, u))
    return u, v


def _smooth_tangents(centers, half=TANGENT_HALF_WINDOW_STATIONS):
    centers = np.asarray(centers, float)
    grad = np.gradient(centers, axis=0)
    grad /= np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-9)
    out = np.zeros_like(centers)
    for i in range(len(centers)):
        lo, hi = max(0, i-half), min(len(centers), i+half+1)
        pts = centers[lo:hi]
        if len(pts) < 3:
            out[i] = grad[i]
            continue
        q = pts - pts.mean(axis=0)
        _, vec = np.linalg.eigh(q.T @ q)
        t = vec[:, -1]
        if float(t @ grad[i]) < 0:
            t = -t
        out[i] = _unit(t)
    return grad, out


class SourceGeometry:
    def __init__(self, meta):
        self.spacing_zyx = np.asarray(meta["spacing_zyx"], float)
        self.spacing_xyz = self.spacing_zyx[::-1]
        self.origin = np.asarray(meta["positions_lps_mm"][0], float)
        iop = np.asarray(meta["image_orientation_patient"], float)
        row, col = iop[:3], iop[3:]
        slc = np.cross(row, col)
        self.D = np.array([[row[0],col[0],slc[0]],[row[1],col[1],slc[1]],[row[2],col[2],slc[2]]], float)
        self.invD = np.linalg.inv(self.D)

    def xyz_to_zyx(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        xyz = ((pts-self.origin) @ self.invD.T) / self.spacing_xyz
        return xyz[:, ::-1]

    def zyx_to_xyz(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        idx_xyz = pts[:, ::-1]
        return self.origin + (idx_xyz*self.spacing_xyz) @ self.D.T


def _load_source(cache):
    cache = Path(cache)
    src = np.load(_req(cache/"series7_int16.npy"), mmap_mode="r")
    meta = _read_json(cache/"series7_int16.json")
    geom = SourceGeometry(meta)
    vv = float(np.prod(geom.spacing_zyx))
    return geom, src, vv


def _sample(geom, src, pts, cval=-1024.0):
    z = geom.xyz_to_zyx(pts)
    return map_coordinates(np.asarray(src), z.T, order=1, mode="constant", cval=float(cval))


def _cmedian(x):
    return np.median(np.stack([np.roll(x,k) for k in range(-2,3)]), axis=0)


def _boundary(geom, src, c, t, n_angles=N_ANGLES, radial_step=RADIAL_STEP_MM, radial_max=LUMEN_MAX_RADIUS_MM):
    u, v = _orth_basis(t)
    th = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    dirs = np.cos(th)[:,None]*u + np.sin(th)[:,None]*v
    center_pts = np.vstack([c,c+.15*u,c-.15*u,c+.15*v,c-.15*v])
    center_hu = float(np.median(_sample(geom, src, center_pts)))
    threshold = float(np.clip(.55*center_hu, 220.0, 500.0))
    rr = np.arange(.2, radial_max+1e-9, radial_step)
    P = c[None,None,:] + dirs[:,None,:]*rr[None,:,None]
    hu = _sample(geom, src, P.reshape(-1,3)).reshape(n_angles, len(rr))
    radii = np.full(n_angles, np.nan)
    for i in range(n_angles):
        b = (hu[i] < threshold) & (rr >= LUMEN_MIN_RADIUS_MM)
        ix = np.flatnonzero(b[:-1] & b[1:])
        if len(ix):
            radii[i] = rr[int(ix[0])]
    valid = np.isfinite(radii)
    vf = float(valid.mean())
    if valid.any():
        med = float(np.median(radii[valid]))
        radii[~valid] = med
        radii = _cmedian(radii)
        radii = np.clip(radii, max(.55,med-.75), min(4.0,med+.75))
        p10,p50,p90 = np.percentile(radii,[10,50,90])
        axis = float(p90/max(p10,1e-6))
    else:
        p10=p50=p90=axis=np.nan
    qc = bool(center_hu >= 200 and vf >= .60 and np.isfinite(axis) and axis <= 2.5)
    return dict(center_hu=center_hu, lumen_threshold_hu=threshold, valid_radial_fraction=vf,
                lumen_radius_p10_mm=float(p10), lumen_radius_median_mm=float(p50),
                lumen_radius_p90_mm=float(p90), lumen_axis_proxy=float(axis),
                station_qc_pass=qc, radii=radii, theta=th, u=u, v=v)


def _candidate_offsets(t):
    u,v = _orth_basis(t)
    out = [(0.0, np.zeros(3))]
    for radius in CENTER_RING_RADII_MM:
        for k in range(CENTER_RING_DIRECTIONS):
            th = 2*np.pi*k/CENTER_RING_DIRECTIONS
            off = radius*(math.cos(th)*u + math.sin(th)*v)
            out.append((float(radius),off))
    return out


def _preview_score(geom, src, c0, t, offset_mm, off, original_hu):
    c = c0 + off
    L = _boundary(geom,src,c,t,n_angles=PREVIEW_ANGLES,radial_step=PREVIEW_RADIAL_STEP_MM,radial_max=2.8)
    if not np.isfinite(L["lumen_axis_proxy"]) or L["center_hu"] < max(200.0,.78*original_hu):
        return -np.inf, L
    axis = max(L["lumen_axis_proxy"],1.0)
    score = 2.0*L["valid_radial_fraction"] - .85*abs(math.log(axis)) + .00045*min(L["center_hu"],900.0) - .28*(offset_mm/CENTER_SEARCH_RADIUS_MM)**2
    return float(score),L


def _refined_lumen(geom, src, c0, t_raw, t_smooth):
    raw = _boundary(geom,src,c0,t_raw)
    base = _boundary(geom,src,c0,t_smooth)
    best = (-np.inf,0.0,np.zeros(3))
    original_hu = max(base["center_hu"],1.0)
    for offset_mm,off in _candidate_offsets(t_smooth):
        score,_ = _preview_score(geom,src,c0,t_smooth,offset_mm,off,original_hu)
        if score > best[0]:
            best=(score,offset_mm,off)
    _,offset_mm,off = best
    c = c0 + off
    final = _boundary(geom,src,c,t_smooth)
    final["refined_center_lps_x_mm"]=float(c[0])
    final["refined_center_lps_y_mm"]=float(c[1])
    final["refined_center_lps_z_mm"]=float(c[2])
    final["center_refine_offset_mm"]=float(offset_mm)
    final["raw_station_qc_pass"]=bool(raw["station_qc_pass"])
    return final


def _join(frozen, extension):
    ext = extension if np.linalg.norm(extension[-1]-frozen[0]) <= np.linalg.norm(extension[0]-frozen[0]) else extension[::-1].copy()
    gap = float(np.linalg.norm(ext[-1]-frozen[0]))
    if gap > 1.5:
        raise RuntimeError(f"Distal extension join gap too large: {gap:.3f} mm")
    ext_len = float(_arc(ext)[-1])
    return np.vstack([ext,frozen[1:]]), ext_len, gap


def _build_geometry(geom, src, combined, ext_len):
    centers, combined_arc = _resample(combined)
    raw_t, smooth_t = _smooth_tangents(centers)
    rows=[]
    for i,(c,a,tr,ts) in enumerate(zip(centers,combined_arc,raw_t,smooth_t)):
        L = _refined_lumen(geom,src,c,tr,ts)
        rows.append({
            "station_index":i,
            "combined_arc_mm":float(a),
            "frozen_arc_mm":float(a-ext_len),
            "lps_x_mm":L["refined_center_lps_x_mm"],
            "lps_y_mm":L["refined_center_lps_y_mm"],
            "lps_z_mm":L["refined_center_lps_z_mm"],
            "tangent_x":float(ts[0]),"tangent_y":float(ts[1]),"tangent_z":float(ts[2]),
            "center_hu":float(L["center_hu"]),
            "lumen_radius_median_mm":float(L["lumen_radius_median_mm"]),
            "lumen_axis_proxy":float(L["lumen_axis_proxy"]),
            "station_qc_pass":bool(L["station_qc_pass"]),
            "center_refine_offset_mm":float(L["center_refine_offset_mm"]),
        })
    return pd.DataFrame(rows)


def _sitk_reference(shape_zyx, geom):
    ref = sitk.Image([int(x) for x in shape_zyx[::-1]], sitk.sitkUInt8)
    ref.SetSpacing(tuple(float(x) for x in geom.spacing_xyz))
    ref.SetOrigin(tuple(float(x) for x in geom.origin))
    ref.SetDirection(tuple(float(x) for x in geom.D.ravel()))
    return ref


def _load_aorta(root, source_shape, geom):
    candidates = [
        root/"RCA_Ostium_TotalSegmentator"/"aorta_series7_totalseg.nii.gz",
        root/"TotalSegmentator_Validation_v2"/"aorta_series7_totalseg.nii.gz",
        root/"TotalSegmentator_Cardiovascular_Cache_v1"/"aorta.nii.gz",
    ]
    p = next((x for x in candidates if x.exists()),None)
    if p is None:
        return None,None
    im = sitk.ReadImage(str(p))
    ref = _sitk_reference(source_shape,geom)
    same = im.GetSize()==ref.GetSize() and np.allclose(im.GetSpacing(),ref.GetSpacing()) and np.allclose(im.GetOrigin(),ref.GetOrigin()) and np.allclose(im.GetDirection(),ref.GetDirection())
    if not same:
        im = sitk.Resample(im,ref,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
    arr = sitk.GetArrayFromImage(im)>0
    return arr,p


def _segment(geometry, lo, hi):
    d = geometry[
        (geometry.frozen_arc_mm >= lo)
        & (geometry.frozen_arc_mm <= hi)
        & geometry.station_qc_pass.astype(bool)
    ].copy().sort_values("frozen_arc_mm")
    if len(d)<2:
        raise RuntimeError(f"Too few LAD PCAT stations in {lo}..{hi}")
    return d


def _voxel_map(geom,src,aorta,segment,max_margin):
    centers = segment[["lps_x_mm","lps_y_mm","lps_z_mm"]].to_numpy(float)
    arcs = segment.frozen_arc_mm.to_numpy(float)
    radii = segment.lumen_radius_median_mm.to_numpy(float)
    max_outer = float(np.max(radii+max_margin))
    pad_mm = 3.0*max_outer + 2.0
    zyx = geom.xyz_to_zyx(centers)
    lo = np.floor(np.min(zyx,axis=0)-pad_mm/geom.spacing_zyx).astype(int)
    hi = np.ceil(np.max(zyx,axis=0)+pad_mm/geom.spacing_zyx).astype(int)+1
    lo = np.maximum(lo,0); hi = np.minimum(hi,np.asarray(src.shape))
    crop = np.asarray(src[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]])
    zz,yy,xx=np.indices(crop.shape)
    gzyx=np.stack([zz+lo[0],yy+lo[1],xx+lo[2]],axis=-1).reshape(-1,3).astype(float)
    gxyz=geom.zyx_to_xyz(gzyx)
    tree=cKDTree(centers)
    dist,idx=tree.query(gxyz,k=1,workers=-1)
    af = aorta[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]].reshape(-1) if aorta is not None else np.zeros(crop.size,bool)
    return dict(hu=crop.reshape(-1).astype(float),dist_mm=dist.astype(float),
                nearest_arc_mm=arcs[idx].astype(float),nearest_lumen_radius_mm=radii[idx].astype(float),
                aorta_flat=af,arc_start_mm=float(arcs.min()),arc_end_mm=float(arcs.max()))


def _compute_pcat(vm,margin,voxel_volume):
    margin=float(margin); hu=vm["hu"]
    outer=vm["nearest_lumen_radius_mm"]+margin
    shell_outer=3.0*outer
    shell=(vm["dist_mm"]>outer)&(vm["dist_mm"]<=shell_outer)&(~vm["aorta_flat"])
    fat=shell&(hu>=FAT_HU_RANGE[0])&(hu<=FAT_HU_RANGE[1])
    vals=hu[fat]
    if len(vals)==0:
        raise RuntimeError("No LAD PCAT fat voxels")
    start=int(math.floor(vm["arc_start_mm"])); stop=int(math.ceil(vm["arc_end_mm"]))
    long_rows=[]
    for b in range(start,stop):
        m=fat&(vm["nearest_arc_mm"]>=b)&(vm["nearest_arc_mm"]<b+1)
        v=hu[m]
        long_rows.append({"arc_start_mm":float(b),"arc_end_mm":float(b+1),"fat_voxels":int(len(v)),"mean_hu":float(np.mean(v)) if len(v) else np.nan})
    return {
        "wall_margin_mm":margin,
        "pcat_mean_hu":float(np.mean(vals)),
        "pcat_median_hu":float(np.median(vals)),
        "pcat_sd_hu":float(np.std(vals)),
        "fat_voxels":int(fat.sum()),
        "fat_volume_ml":float(fat.sum()*voxel_volume/1000.0),
        "shell_voxels":int(shell.sum()),
        "shell_volume_ml":float(shell.sum()*voxel_volume/1000.0),
        "fat_fraction":float(fat.sum()/max(1,shell.sum())),
        "longitudinal":pd.DataFrame(long_rows),
    }


def _run_segment(geom,src,aorta,geometry,label,range_mm,voxel_volume):
    seg=_segment(geometry,*range_mm)
    vm=_voxel_map(geom,src,aorta,seg,max(MARGINS_MM))
    results=[_compute_pcat(vm,m,voxel_volume) for m in MARGINS_MM]
    primary=next(r for r in results if abs(r["wall_margin_mm"]-PRIMARY_WALL_MARGIN_MM)<1e-12)
    repeat=_compute_pcat(vm,PRIMARY_WALL_MARGIN_MM,voxel_volume)
    exact=bool(primary["pcat_mean_hu"]==repeat["pcat_mean_hu"] and primary["fat_voxels"]==repeat["fat_voxels"] and primary["shell_voxels"]==repeat["shell_voxels"])
    long=primary["longitudinal"]
    good=long.fat_voxels>=MIN_BIN_FAT_VOXELS
    summary={
        "segment":label,
        "frozen_arc_start_mm":float(seg.frozen_arc_mm.min()),
        "frozen_arc_end_mm":float(seg.frozen_arc_mm.max()),
        "segment_length_mm":float(seg.frozen_arc_mm.max()-seg.frozen_arc_mm.min()),
        "station_count":int(len(seg)),
        "station_qc_fraction":float(seg.station_qc_pass.mean()),
        "primary_wall_margin_mm":PRIMARY_WALL_MARGIN_MM,
        "pcat_mean_hu":primary["pcat_mean_hu"],
        "pcat_median_hu":primary["pcat_median_hu"],
        "pcat_sd_hu":primary["pcat_sd_hu"],
        "fat_voxels":primary["fat_voxels"],
        "fat_volume_ml":primary["fat_volume_ml"],
        "shell_volume_ml":primary["shell_volume_ml"],
        "fat_fraction":primary["fat_fraction"],
        "longitudinal_bin_count":int(len(long)),
        "longitudinal_bins_ge_min_fat_voxels":int(good.sum()),
        "longitudinal_bin_coverage_fraction":float(good.mean()) if len(good) else 0.0,
        "sensitivity_mean_hu_min":float(min(r["pcat_mean_hu"] for r in results)),
        "sensitivity_mean_hu_max":float(max(r["pcat_mean_hu"] for r in results)),
        "sensitivity_range_hu":float(max(r["pcat_mean_hu"] for r in results)-min(r["pcat_mean_hu"] for r in results)),
        "exact_repeat_pass":exact,
    }
    sens=pd.DataFrame([{k:v for k,v in r.items() if k!="longitudinal"} for r in results])
    return summary,primary,sens


def _plot_longitudinal(results,out):
    fig,ax=plt.subplots(figsize=(11,5.5))
    for label,r in results.items():
        d=r["longitudinal"]
        ax.plot((d.arc_start_mm+d.arc_end_mm)/2,d.mean_hu,marker="o",label=label)
    ax.set_xlabel("Frozen-LAD arc coordinate (mm)")
    ax.set_ylabel("PCAT mean HU")
    ax.set_title("LAD direct PCAT longitudinal profiles")
    ax.legend()
    fig.tight_layout();fig.savefig(out,dpi=170);plt.close(fig)


def _plot_sensitivity(sens,out):
    fig,ax=plt.subplots(figsize=(8,5.5))
    for label,d in sens.items():
        ax.plot(d.wall_margin_mm,d.pcat_mean_hu,marker="o",label=label)
    ax.axvline(PRIMARY_WALL_MARGIN_MM,linestyle="--",linewidth=1)
    ax.set_xlabel("Modeled outer-wall margin (mm)")
    ax.set_ylabel("PCAT mean HU")
    ax.set_title("LAD PCAT geometry sensitivity")
    ax.legend()
    fig.tight_layout();fig.savefig(out,dpi=170);plt.close(fig)


def _plot_geometry(geometry,out):
    fig,ax=plt.subplots(figsize=(11,5.5))
    ax.plot(geometry.frozen_arc_mm,geometry.lumen_axis_proxy,label="refined lumen axis proxy")
    ax.axhline(2.5,linestyle="--",linewidth=1)
    ax.axvspan(*FROZEN_SEGMENT_RANGE_MM,alpha=.06,label="frozen LAD PCAT segment")
    ax.axvspan(*DISTAL_SEGMENT_RANGE_MM,alpha=.06,label="distal continuation PCAT segment")
    ax.set_xlabel("Frozen-LAD arc coordinate (mm)")
    ax.set_ylabel("p90/p10 lumen axis proxy")
    ax.set_title("LAD source-plane geometry used for PCAT")
    ax.legend()
    fig.tight_layout();fig.savefig(out,dpi=170);plt.close(fig)


def synthetic_self_test():
    p=np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
    e=np.array([[-2.,0.,0.],[-1.,0.,0.],[0.,0.,0.]])
    combined,ext_len,gap=_join(p,e)
    assert np.isclose(ext_len,2.0) and gap<1e-9
    assert len(combined)==5
    return {"ok":True,"extension_length_mm":ext_len}


def run(drive_root="/content/drive/MyDrive/OpenPlaque",output_dir=None):
    root=Path(drive_root)
    out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME
    out.mkdir(parents=True,exist_ok=True)
    _write_json(out/"run_state.json",{"status":"STARTED","algorithm":ALGORITHM,"baseline":BASELINE})

    master=_read_json(root/MASTER)
    bidir=_read_json(root/DISTAL_BIDIR_SUMMARY)
    if master.get("status")!=EXPECTED_MASTER_STATUS:
        raise RuntimeError("Frozen master prerequisite failed")
    if bidir.get("status")!=EXPECTED_BIDIR_STATUS:
        raise RuntimeError("Validated distal LAD prerequisite failed")

    geom,src,vv=_load_source(root/SOURCE_CACHE)
    frozen=_load_path(root/FROZEN_LAD)
    ext=_load_path(root/DISTAL_EXTENSION)
    combined,ext_len,join_gap=_join(frozen,ext)
    geometry=_build_geometry(geom,src,combined,ext_len)
    geometry.to_csv(out/"LAD_PCAT_source_geometry.csv",index=False)

    frozen_qc=float(geometry[(geometry.frozen_arc_mm>=FROZEN_SEGMENT_RANGE_MM[0])&(geometry.frozen_arc_mm<=FROZEN_SEGMENT_RANGE_MM[1])].station_qc_pass.mean())
    distal_qc=float(geometry[(geometry.frozen_arc_mm>=DISTAL_SEGMENT_RANGE_MM[0])&(geometry.frozen_arc_mm<=DISTAL_SEGMENT_RANGE_MM[1])].station_qc_pass.mean())

    aorta,aorta_path=_load_aorta(root,src.shape,geom)

    segment_defs={
        "frozen_LAD":FROZEN_SEGMENT_RANGE_MM,
        "validated_distal_continuation":DISTAL_SEGMENT_RANGE_MM,
    }
    summaries={}; primaries={}; sensitivities={}
    for label,rng in segment_defs.items():
        s,p,se=_run_segment(geom,src,aorta,geometry,label,rng,vv)
        summaries[label]=s;primaries[label]=p;sensitivities[label]=se
        se.assign(segment=label).to_csv(out/f"{label}_PCAT_margin_sensitivity.csv",index=False)
        p["longitudinal"].assign(segment=label).to_csv(out/f"{label}_PCAT_longitudinal.csv",index=False)

    frozen_pass=bool(
        summaries["frozen_LAD"]["segment_length_mm"]>=MIN_FROZEN_SEGMENT_LENGTH_MM
        and summaries["frozen_LAD"]["fat_voxels"]>=MIN_FAT_VOXELS
        and summaries["frozen_LAD"]["longitudinal_bin_coverage_fraction"]>=MIN_LONGITUDINAL_COVERAGE
        and summaries["frozen_LAD"]["exact_repeat_pass"]
        and frozen_qc>=.95
    )
    distal_pass=bool(
        summaries["validated_distal_continuation"]["segment_length_mm"]>=MIN_DISTAL_SEGMENT_LENGTH_MM
        and summaries["validated_distal_continuation"]["fat_voxels"]>=MIN_FAT_VOXELS
        and summaries["validated_distal_continuation"]["longitudinal_bin_coverage_fraction"]>=MIN_LONGITUDINAL_COVERAGE
        and summaries["validated_distal_continuation"]["exact_repeat_pass"]
        and distal_qc>=.95
    )
    status=STATUS_PASS if frozen_pass and distal_pass else STATUS_FAIL

    pd.DataFrame(list(summaries.values())).to_csv(out/"LAD_PCAT_segment_summary.csv",index=False)
    _plot_longitudinal(primaries,out/"01_LAD_PCAT_longitudinal.png")
    _plot_sensitivity(sensitivities,out/"02_LAD_PCAT_margin_sensitivity.png")
    _plot_geometry(geometry,out/"03_LAD_PCAT_geometry_QC.png")

    summary={
        "status":status,
        "algorithm":ALGORITHM,
        "baseline_commit":BASELINE,
        "master_status":master.get("status"),
        "master_modified":False,
        "frozen_lad_length_mm":float(_arc(frozen)[-1]),
        "distal_extension_length_mm":ext_len,
        "join_gap_mm":join_gap,
        "source_voxel_volume_mm3":vv,
        "aorta_exclusion_available":bool(aorta is not None),
        "aorta_exclusion_path":str(aorta_path) if aorta_path is not None else None,
        "geometry_qc":{"frozen_segment_qc_fraction":frozen_qc,"distal_segment_qc_fraction":distal_qc},
        "segments":summaries,
        "technical_feasibility":{"frozen_LAD":frozen_pass,"validated_distal_continuation":distal_pass},
        "is_proprietary_fai":False,
        "is_validated_clinical_biomarker":False,
        "scientific_boundary":(
            "Direct LAD PCAT attenuation is measured on two research segments using the same -190 to -30 HU adipose window and +0.75 mm nominal modeled outer-wall margin as the locked RCA prototype. "
            "The frozen-LAD segment excludes 1 mm at each end of the accepted 24.997-mm centerline. The distal continuation is independently anatomy-validated but not part of the frozen master. "
            "These are research direct-attenuation measurements, not proprietary FAI and not independent clinical validation."
        ),
    }
    _write_json(out/"summary.json",summary)
    _write_json(out/"input_provenance.json",{
        "source_cache":str(root/SOURCE_CACHE),
        "frozen_lad":str(root/FROZEN_LAD),
        "distal_extension":str(root/DISTAL_EXTENSION),
        "distal_bidirectional_summary":str(root/DISTAL_BIDIR_SUMMARY),
        "master":str(root/MASTER),
    })

    report=out/"OPENPLAQUE_LAD_SOURCE_SPACE_PCAT_FEASIBILITY_REPORT.html"
    report.write_text(
        "<html><body><h1>OpenPlaque LAD Source-Space PCAT Feasibility v1</h1>"
        f"<p><b>Status:</b> {status}</p>"
        f"<p>Frozen-LAD direct PCAT: {summaries['frozen_LAD']['pcat_mean_hu']:.2f} HU over {summaries['frozen_LAD']['fat_voxels']:,} fat voxels.</p>"
        f"<p>Validated distal-continuation direct PCAT: {summaries['validated_distal_continuation']['pcat_mean_hu']:.2f} HU over {summaries['validated_distal_continuation']['fat_voxels']:,} fat voxels.</p>"
        "<p><b>Boundary:</b> research direct attenuation only; not proprietary FAI. Distal continuation is not part of the frozen master.</p>"
        '<img src="01_LAD_PCAT_longitudinal.png" style="max-width:100%">'
        '<img src="02_LAD_PCAT_margin_sensitivity.png" style="max-width:100%">'
        '<img src="03_LAD_PCAT_geometry_QC.png" style="max-width:100%">'
        "</body></html>",encoding="utf-8"
    )

    _write_json(out/"run_state.json",{"status":"COMPLETE","result_status":status,"algorithm":ALGORITHM,"baseline":BASELINE})
    zpath=out/"OPENPLAQUE_LAD_SOURCE_SPACE_PCAT_FEASIBILITY_RESULTS.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p!=zpath:
                z.write(p,p.name)
    return {"summary":summary,"report":str(report),"zip":str(zpath)}
