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
from scipy.ndimage import map_coordinates
from scipy.stats import rankdata, spearmanr

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "cross-vessel-lumen-frame-refinement-plaque-rescue-v1.0"
OUTPUT_DIRNAME = "Cross_Vessel_Lumen_Frame_Refinement_Plaque_Rescue_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
RCA_CENTERLINE = Path("Cache/Source_Volume_Coronary_Centerlines/RCA_source_centerline.csv")
FROZEN_LAD = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
DISTAL_EXTENSION = Path("LAD_Distal_Endpoint_Continuation_v1/best_independent_distal_extension.csv")
DISTAL_BIDIR_SUMMARY = Path("LAD_Distal_Bidirectional_Validation_v1/summary.json")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
RCA_PRIOR = Path("Longitudinal_Plaque_PCAT_Fusion_v1/RCA_source_longitudinal_plaque_profile_1mm.csv")
LAD_PRIOR = Path("Longitudinal_Plaque_PCAT_Fusion_v1/LAD_source_longitudinal_plaque_profile_1mm.csv")
PREV_RCA_SUMMARY = Path("RCA_Source_Space_Plaque_Excess_Specificity_v1/summary.json")
PREV_LAD_SUMMARY = Path("LAD_Source_Space_Plaque_Transfer_Validation_v1/summary.json")

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

SHELLS = (0.75, 1.0, 1.25, 1.5)
NOMINAL_SHELL = 1.0
RCA_REFERENCE_ARC = (20.0, 50.0)
RCA_HOLDOUT_ARC = (0.0, 20.0)
REFERENCE_RESIDUAL_QUANTILE = 0.90
HUBER_K = 1.35
COMPONENTS = (
    "low_attenuation_mm3",
    "noncalcified_mm3",
    "mixed_intermediate_mm3",
    "calcified_mm3",
)

MIN_RCA_QC = 0.98
MIN_LAD_FROZEN_QC = 0.90
MIN_LAD_EXTENSION_QC = 0.85
MIN_POSITIVE_BINS = 4
MIN_NEGATIVE_BINS = 5
MIN_STRICT_BINS = 2
MIN_MAJORITY_AUC = 0.80
MIN_STRICT_AUC = 0.90
MIN_POS_NEG_RATIO = 2.00
MIN_CROSS_SHELL_SPEARMAN = 0.80

STATUS_RCA_FAIL = "FRAME_REFINEMENT_RCA_CONTROL_FAILED"
STATUS_LAD_GEOM_FAIL = "FRAME_REFINEMENT_LAD_GEOMETRY_NOT_RESCUED"
STATUS_LAD_SPEC_FAIL = "FRAME_REFINEMENT_LAD_GEOMETRY_RESCUED_SPECIFICITY_NOT_TRANSFERRED"
STATUS_PASS = "FRAME_REFINEMENT_LAD_DEVELOPMENTAL_TRANSFER_PASS"


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
    for cols in (("lps_x_mm", "lps_y_mm", "lps_z_mm"), ("x_mm", "y_mm", "z_mm")):
        if all(c in d.columns for c in cols):
            return d[list(cols)].to_numpy(float)
    raise ValueError(f"No LPS coordinate columns in {p}: {list(d.columns)}")


def _arc(p):
    p = np.asarray(p, float)
    if len(p) <= 1:
        return np.zeros(len(p))
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def _interp(p, q):
    p = np.asarray(p, float)
    a = _arc(p)
    q = np.asarray(q, float)
    return np.column_stack([np.interp(q, a, p[:, k]) for k in range(3)])


def _resample(p, step=ARC_STEP_MM):
    p = np.asarray(p, float)
    a = _arc(p)
    if len(p) < 2 or a[-1] <= 0:
        return p.copy(), a
    q = np.arange(0.0, float(a[-1]) + 1e-9, step)
    if q[-1] < a[-1] - 1e-6:
        q = np.r_[q, a[-1]]
    return _interp(p, q), q


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


def _orth_basis(t):
    t = _unit(t)
    axes = np.eye(3)
    seed = axes[np.argmin(np.abs(axes @ t))]
    u = _unit(np.cross(t, seed))
    v = _unit(np.cross(t, u))
    return u, v


def _arc_weights(q):
    q = np.asarray(q, float)
    if len(q) == 1:
        return np.ones(1)
    w = np.empty(len(q), float)
    w[0] = 0.5 * (q[1] - q[0])
    w[-1] = 0.5 * (q[-1] - q[-2])
    if len(q) > 2:
        w[1:-1] = 0.5 * (q[2:] - q[:-2])
    return w


def _angle_deg(a, b):
    a, b = _unit(a), _unit(b)
    return float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), -1.0, 1.0))))


class SourceGeometry:
    def __init__(self, meta):
        self.spacing_zyx = np.asarray(meta["spacing_zyx"], float)
        self.spacing_xyz = self.spacing_zyx[::-1]
        self.origin = np.asarray(meta["positions_lps_mm"][0], float)
        iop = np.asarray(meta["image_orientation_patient"], float)
        row, col = iop[:3], iop[3:]
        slc = np.cross(row, col)
        self.D = np.array(
            [[row[0], col[0], slc[0]],
             [row[1], col[1], slc[1]],
             [row[2], col[2], slc[2]]], float
        )
        self.invD = np.linalg.inv(self.D)

    def xyz_to_zyx(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        xyz = ((pts - self.origin) @ self.invD.T) / self.spacing_xyz
        return xyz[:, ::-1]


def _load_source(cache):
    cache = Path(cache)
    src = np.load(_req(cache / "series7_int16.npy"), mmap_mode="r")
    meta = _read_json(cache / "series7_int16.json")
    geom = SourceGeometry(meta)
    vv = float(np.prod(geom.spacing_zyx))
    if not (0.001 < vv < 0.5):
        raise RuntimeError(f"Unexpected Series-7 voxel volume {vv}")
    return geom, src, vv


def _sample(geom, src, pts, cval=-1024.0):
    z = geom.xyz_to_zyx(pts)
    return map_coordinates(np.asarray(src), z.T, order=1, mode="constant", cval=float(cval))


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
        cov = q.T @ q
        val, vec = np.linalg.eigh(cov)
        t = vec[:, int(np.argmax(val))]
        if float(t @ grad[i]) < 0:
            t = -t
        out[i] = _unit(t)
    return grad, out


def _cmedian(x):
    return np.median(np.stack([np.roll(x, k) for k in range(-2, 3)]), axis=0)


def _boundary(geom, src, c, t, n_angles=N_ANGLES, radial_step=RADIAL_STEP_MM, radial_max=LUMEN_MAX_RADIUS_MM):
    u, v = _orth_basis(t)
    th = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    dirs = np.cos(th)[:, None]*u + np.sin(th)[:, None]*v
    center_pts = np.vstack([c, c+.15*u, c-.15*u, c+.15*v, c-.15*v])
    center_hu = float(np.median(_sample(geom, src, center_pts)))
    threshold = float(np.clip(.55*center_hu, 220.0, 500.0))
    rr = np.arange(.2, radial_max + 1e-9, radial_step)
    P = c[None, None, :] + dirs[:, None, :]*rr[None, :, None]
    hu = _sample(geom, src, P.reshape(-1, 3)).reshape(n_angles, len(rr))
    r = np.full(n_angles, np.nan)
    for i in range(n_angles):
        b = (hu[i] < threshold) & (rr >= LUMEN_MIN_RADIUS_MM)
        ix = np.flatnonzero(b[:-1] & b[1:])
        if len(ix):
            r[i] = rr[int(ix[0])]
    valid = np.isfinite(r)
    vf = float(valid.mean())
    if valid.any():
        med0 = float(np.median(r[valid]))
        r[~valid] = med0
        r = _cmedian(r)
        r = np.clip(r, med0-.75, med0+.75)
        r = np.clip(r, .55, 4.0)
        p10, p50, p90 = np.percentile(r, [10, 50, 90])
        axis = float(p90 / max(p10, 1e-6))
        area = float(.5*np.sum(r*r)*(2*np.pi/n_angles))
    else:
        p10=p50=p90=axis=area=np.nan
    qc = bool(center_hu >= 200 and vf >= .60 and np.isfinite(axis) and axis <= 2.5)
    return dict(
        center_hu=center_hu, lumen_threshold_hu=threshold, valid_radial_fraction=vf,
        lumen_radius_p10_mm=float(p10), lumen_radius_median_mm=float(p50),
        lumen_radius_p90_mm=float(p90), lumen_axis_proxy=float(axis),
        lumen_area_mm2=float(area), station_qc_pass=qc,
        radii=r, theta=th, u=u, v=v,
    )


def _candidate_offsets(t):
    u, v = _orth_basis(t)
    out = [(0.0, np.zeros(3))]
    for radius in CENTER_RING_RADII_MM:
        for k in range(CENTER_RING_DIRECTIONS):
            th = 2*np.pi*k/CENTER_RING_DIRECTIONS
            off = radius*(math.cos(th)*u + math.sin(th)*v)
            out.append((float(radius), off))
    return out


def _preview_score(geom, src, c0, t, offset_mm, off, original_hu):
    c = c0 + off
    L = _boundary(
        geom, src, c, t, n_angles=PREVIEW_ANGLES,
        radial_step=PREVIEW_RADIAL_STEP_MM, radial_max=2.8
    )
    if not np.isfinite(L["lumen_axis_proxy"]) or L["center_hu"] < max(200.0, .78*original_hu):
        return -np.inf, L
    axis = max(L["lumen_axis_proxy"], 1.0)
    score = (
        2.00*L["valid_radial_fraction"]
        - 0.85*abs(math.log(axis))
        + 0.00045*min(L["center_hu"], 900.0)
        - 0.28*(offset_mm/CENTER_SEARCH_RADIUS_MM)**2
    )
    return float(score), L


def _refined_lumen(geom, src, c0, t_raw, t_smooth):
    raw = _boundary(geom, src, c0, t_raw)
    base = _boundary(geom, src, c0, t_smooth)
    best = (-np.inf, 0.0, np.zeros(3), base)
    original_hu = max(base["center_hu"], 1.0)
    for offset_mm, off in _candidate_offsets(t_smooth):
        score, preview = _preview_score(geom, src, c0, t_smooth, offset_mm, off, original_hu)
        if score > best[0]:
            best = (score, offset_mm, off, preview)
    _, offset_mm, off, _ = best
    c = c0 + off
    final = _boundary(geom, src, c, t_smooth)
    final["refined_center_lps_x_mm"] = float(c[0])
    final["refined_center_lps_y_mm"] = float(c[1])
    final["refined_center_lps_z_mm"] = float(c[2])
    final["center_refine_offset_mm"] = float(offset_mm)
    final["raw_center_hu"] = float(raw["center_hu"])
    final["raw_lumen_axis_proxy"] = float(raw["lumen_axis_proxy"])
    final["raw_station_qc_pass"] = bool(raw["station_qc_pass"])
    final["smoothed_unshifted_axis_proxy"] = float(base["lumen_axis_proxy"])
    final["tangent_refine_angle_deg"] = _angle_deg(t_raw, t_smooth)
    return final


def _integrate_shell(geom, src, c, L, shell, ds):
    th = L["theta"]
    dirs = np.cos(th)[:, None]*L["u"] + np.sin(th)[:, None]*L["v"]
    off = np.arange(RADIAL_STEP_MM/2.0, shell, RADIAL_STEP_MM)
    if len(off) == 0:
        off = np.array([shell/2.0])
    r = L["radii"][:, None] + off[None, :]
    P = c[None, None, :] + dirs[:, None, :]*r[..., None]
    hu = _sample(geom, src, P.reshape(-1, 3)).reshape(r.shape)
    area = r*RADIAL_STEP_MM*(2*np.pi/N_ANGLES)
    vol = area*float(ds)
    finite = np.isfinite(hu)
    masks = {
        "fatlike_excluded_mm3": finite & (hu < -30),
        "low_attenuation_mm3": finite & (hu >= -30) & (hu < 30),
        "noncalcified_mm3": finite & (hu >= 30) & (hu < 130),
        "mixed_intermediate_mm3": finite & (hu >= 130) & (hu < 350),
        "calcified_mm3": finite & (hu >= 350),
    }
    vals = {k: float(vol[m].sum()) for k,m in masks.items()}
    vals["shell_volume_mm3"] = float(vol[finite].sum())
    vals["total_plaque_proxy_mm3"] = sum(vals[c] for c in COMPONENTS)
    return vals


def _quantify_path(geom, src, path, vessel, frozen_arc_offset=0.0):
    centers, arcs = _resample(path)
    raw_t, smooth_t = _smooth_tangents(centers)
    weights = _arc_weights(arcs)
    rows=[]
    for i,(c,tr,ts,a,ds) in enumerate(zip(centers,raw_t,smooth_t,arcs,weights)):
        L = _refined_lumen(geom,src,c,tr,ts)
        rc = np.array([L["refined_center_lps_x_mm"],L["refined_center_lps_y_mm"],L["refined_center_lps_z_mm"]])
        base={k:v for k,v in L.items() if k not in ("radii","theta","u","v")}
        frozen_arc = float(a + frozen_arc_offset)
        region = "distal_extension" if vessel=="LAD" and frozen_arc < -0.25 else ("frozen_lad" if vessel=="LAD" else "rca")
        for shell in SHELLS:
            if L["station_qc_pass"]:
                comp=_integrate_shell(geom,src,rc,L,shell,ds)
            else:
                comp={k:np.nan for k in ("shell_volume_mm3","fatlike_excluded_mm3","low_attenuation_mm3",
                                         "noncalcified_mm3","mixed_intermediate_mm3","calcified_mm3","total_plaque_proxy_mm3")}
            rows.append({
                "vessel":vessel,"station_index":i,"arc_mm":float(a),"frozen_arc_mm":frozen_arc,
                "integration_ds_mm":float(ds),"shell_thickness_mm":float(shell),"region":region,**base,**comp
            })
    return pd.DataFrame(rows),centers,raw_t,smooth_t


def _design(df):
    r=df["lumen_radius_median_mm"].to_numpy(float)
    hu=df["center_hu"].to_numpy(float)/1000.0
    return np.column_stack([np.ones(len(df)),r,r*r,hu])


def _huber_fit(X,y,k=HUBER_K,n_iter=40):
    X,y=np.asarray(X,float),np.asarray(y,float)
    ok=np.all(np.isfinite(X),axis=1)&np.isfinite(y)
    X,y=X[ok],y[ok]
    if len(y)<X.shape[1]+4:
        raise RuntimeError("Insufficient finite stations for Huber fit")
    beta=np.linalg.lstsq(X,y,rcond=None)[0]
    for _ in range(n_iter):
        resid=y-X@beta
        med=float(np.median(resid)); mad=float(np.median(np.abs(resid-med)))
        scale=max(1.4826*mad,1e-6)
        a=np.abs(resid-med)/scale
        w=np.ones_like(a); hi=a>k; w[hi]=k/np.maximum(a[hi],1e-12)
        sw=np.sqrt(w)
        new=np.linalg.lstsq(X*sw[:,None],y*sw,rcond=None)[0]
        if np.linalg.norm(new-beta)<1e-10:
            beta=new; break
        beta=new
    return beta,y-X@beta


def _fit_rca_models(stations):
    models=[]
    for shell in SHELLS:
        d=stations[np.isclose(stations.shell_thickness_mm,shell)&stations.station_qc_pass.astype(bool)].copy()
        ref=(d.arc_mm>=RCA_REFERENCE_ARC[0])&(d.arc_mm<RCA_REFERENCE_ARC[1])
        if int(ref.sum())<25:
            raise RuntimeError(f"Too few refined RCA reference stations for shell {shell}: {int(ref.sum())}")
        X=_design(d)
        for c in COMPONENTS:
            frac=d[c].to_numpy(float)/np.maximum(d.shell_volume_mm3.to_numpy(float),1e-9)
            beta,resid=_huber_fit(X[ref],frac[ref])
            q=float(np.quantile(resid[np.isfinite(resid)],REFERENCE_RESIDUAL_QUANTILE))
            models.append({
                "shell_thickness_mm":float(shell),"component":c,
                "beta_intercept":float(beta[0]),"beta_lumen_radius":float(beta[1]),
                "beta_lumen_radius_sq":float(beta[2]),"beta_center_hu_per_1000":float(beta[3]),
                "reference_residual_p90":q,"reference_station_count":int(ref.sum())
            })
    return pd.DataFrame(models)


def _apply_models(stations,models):
    out=[]
    for shell in SHELLS:
        d=stations[np.isclose(stations.shell_thickness_mm,shell)].copy()
        X=_design(d)
        for c in COMPONENTS:
            m=models[np.isclose(models.shell_thickness_mm,shell)&(models.component==c)]
            if len(m)!=1:
                raise RuntimeError(f"Missing model {shell} {c}")
            m=m.iloc[0]
            beta=np.array([m.beta_intercept,m.beta_lumen_radius,m.beta_lumen_radius_sq,m.beta_center_hu_per_1000],float)
            frac=d[c].to_numpy(float)/np.maximum(d.shell_volume_mm3.to_numpy(float),1e-9)
            threshold=X@beta+float(m.reference_residual_p90)
            excess=np.maximum(frac-threshold,0.0)*d.shell_volume_mm3.to_numpy(float)
            stem=c.replace("_mm3","")
            d[f"expected_p90_{stem}_fraction"]=threshold
            d[f"excess_{stem}_mm3"]=excess
        ex=[f"excess_{c.replace('_mm3','')}_mm3" for c in COMPONENTS]
        d["excess_total_plaque_proxy_mm3"]=d[ex].sum(axis=1)
        out.append(d)
    return pd.concat(out,ignore_index=True)


def _profile_1mm(scored,shell,vessel):
    d=scored[np.isclose(scored.shell_thickness_mm,shell)&scored.station_qc_pass.astype(bool)].copy()
    coord=d.frozen_arc_mm if vessel=="LAD" else d.arc_mm
    d["profile_arc_mm"]=coord
    d["arc_start_mm"]=np.floor(d.profile_arc_mm.to_numpy(float)).astype(float)
    rows=[]
    excols=[f"excess_{c.replace('_mm3','')}_mm3" for c in COMPONENTS]
    for a,g in d.groupby("arc_start_mm",sort=True):
        row={"shell_thickness_mm":float(shell),"arc_start_mm":float(a),"arc_end_mm":float(a+1),
             "station_count":int(len(g)),"station_qc_fraction":float(g.station_qc_pass.mean()),
             "raw_total_plaque_proxy_mm3":float(g.total_plaque_proxy_mm3.sum())}
        if vessel=="LAD":
            row["region"]="distal_extension" if a<0 else "frozen_lad"
        for c in excols: row[c]=float(g[c].sum())
        row["excess_total_plaque_proxy_mm3"]=float(g.excess_total_plaque_proxy_mm3.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _auc(y,score):
    y=np.asarray(y,int); score=np.asarray(score,float)
    ok=np.isfinite(score)&np.isin(y,[0,1]); y,score=y[ok],score[ok]
    n1,n0=int((y==1).sum()),int((y==0).sum())
    if not n1 or not n0:return np.nan
    r=rankdata(score,method="average")
    return float((r[y==1].sum()-n1*(n1+1)/2.0)/(n1*n0))


def _validation_metrics(profile,prior,arc_lo,arc_hi):
    z=profile.merge(prior,on=["arc_start_mm","arc_end_mm"],how="inner")
    z=z[(z.arc_start_mm>=arc_lo)&(z.arc_start_mm<arc_hi)].copy()
    pos=z.majority_3plus_signal.fillna(False).astype(bool)
    neg=z.mapped_native_vote_sum.fillna(0).to_numpy(float)==0
    strict=z.strict_5of5_signal.fillna(False).astype(bool)
    score=z.excess_total_plaque_proxy_mm3.to_numpy(float)
    out={"usable":bool(int(pos.sum())>=MIN_POSITIVE_BINS and int(neg.sum())>=MIN_NEGATIVE_BINS),
         "matched_bins":int(len(z)),"positive_bins":int(pos.sum()),"strict_5of5_bins":int(strict.sum()),
         "vote_free_negative_bins":int(neg.sum())}
    if not out["usable"]:return z,out
    out["majority_vs_vote_free_auc"]=_auc(np.r_[np.ones(int(pos.sum())),np.zeros(int(neg.sum()))],
                                          np.r_[score[pos],score[neg]])
    out["strict5_vs_vote_free_auc"]=_auc(np.r_[np.ones(int(strict.sum())),np.zeros(int(neg.sum()))],
                                         np.r_[score[strict],score[neg]]) if int(strict.sum())>=MIN_STRICT_BINS else np.nan
    mp=float(np.median(score[pos])); mn=float(np.median(score[neg]))
    out["positive_median_excess_mm3"]=mp; out["negative_median_excess_mm3"]=mn
    out["positive_negative_median_ratio"]=mp/max(mn,1e-6)
    out["spearman_excess_vs_vote_ge3"]=float(spearmanr(z.excess_total_plaque_proxy_mm3,z.mapped_native_voxels_vote_ge3).statistic)
    return z,out


def _cross_shell(profiles,lo,hi):
    nom=profiles[NOMINAL_SHELL]
    nom=nom[(nom.arc_start_mm>=lo)&(nom.arc_start_mm<hi)].set_index("arc_start_mm")["excess_total_plaque_proxy_mm3"]
    rows=[]
    for shell,p in profiles.items():
        y=p[(p.arc_start_mm>=lo)&(p.arc_start_mm<hi)].set_index("arc_start_mm")["excess_total_plaque_proxy_mm3"].reindex(nom.index)
        ok=nom.notna()&y.notna()
        rho=float(spearmanr(nom[ok],y[ok]).statistic) if int(ok.sum())>=5 else np.nan
        rows.append({"shell_thickness_mm":float(shell),"profile_spearman_vs_nominal":rho})
    d=pd.DataFrame(rows)
    f=d[~np.isclose(d.shell_thickness_mm,NOMINAL_SHELL)].profile_spearman_vs_nominal
    f=f[np.isfinite(f)]
    return d,float(f.min()) if len(f) else np.nan


def _passes_specificity(m,min_shell):
    return bool(m.get("usable",False)
        and m.get("majority_vs_vote_free_auc",-np.inf)>=MIN_MAJORITY_AUC
        and (m.get("strict_5of5_bins",0)<MIN_STRICT_BINS or m.get("strict5_vs_vote_free_auc",-np.inf)>=MIN_STRICT_AUC)
        and m.get("positive_negative_median_ratio",-np.inf)>=MIN_POS_NEG_RATIO
        and np.isfinite(min_shell) and min_shell>=MIN_CROSS_SHELL_SPEARMAN)


def _join_frozen_and_extension(frozen,extension,max_join_mm=1.5):
    frozen=np.asarray(frozen,float); extension=np.asarray(extension,float)
    d0=min(float(np.linalg.norm(extension[0]-frozen[0])),float(np.linalg.norm(extension[-1]-frozen[0])))
    if d0>max_join_mm: raise RuntimeError(f"Distal extension does not join frozen LAD arc-0 endpoint: {d0:.3f} mm")
    ext=extension if np.linalg.norm(extension[-1]-frozen[0])<=np.linalg.norm(extension[0]-frozen[0]) else extension[::-1].copy()
    gap=float(np.linalg.norm(ext[-1]-frozen[0]))
    if gap>max_join_mm: raise RuntimeError(f"LAD extension join gap {gap:.3f} mm")
    elen=float(_arc(ext)[-1]); return np.vstack([ext,frozen[1:]]),elen,gap


def _plot_axis_diagnostics(stations,out,title):
    d=stations[np.isclose(stations.shell_thickness_mm,NOMINAL_SHELL)].copy()
    x=d.frozen_arc_mm if "LAD" in title else d.arc_mm
    fig,ax=plt.subplots(figsize=(12,5.5))
    ax.plot(x,d.raw_lumen_axis_proxy,label="original frame axis proxy",alpha=.6)
    ax.plot(x,d.lumen_axis_proxy,label="refined frame axis proxy",linewidth=2)
    ax.axhline(2.5,linestyle="--",linewidth=1,label="QC limit")
    ax2=ax.twinx(); ax2.plot(x,d.center_refine_offset_mm,label="center shift",alpha=.35)
    ax.set_xlabel("arc (mm)"); ax.set_ylabel("p90/p10 radial boundary proxy"); ax2.set_ylabel("center refinement (mm)")
    lines=ax.get_lines()+ax2.get_lines(); ax.legend(lines,[l.get_label() for l in lines],loc="upper right")
    ax.set_title(title); fig.tight_layout(); fig.savefig(out,dpi=170); plt.close(fig)


def _plot_transfer(profile,matched,extension_length,out):
    fig,ax=plt.subplots(figsize=(12,6))
    ax.plot(profile.arc_start_mm+.5,profile.excess_total_plaque_proxy_mm3,label="refined-frame LAD excess",linewidth=2)
    ax.axvline(0,linestyle="--",linewidth=1); ax.axvspan(-extension_length,0,alpha=.06)
    ax.set_xlabel("LAD frozen-axis arc (mm; extension negative)"); ax.set_ylabel("Excess proxy per 1-mm bin (mm³)")
    ax2=ax.twinx()
    if len(matched): ax2.step(matched.arc_start_mm+.5,matched.mapped_native_voxels_vote_ge3.fillna(0),where="mid",alpha=.45,label="prior 3+/5 vote voxels")
    ax2.set_ylabel("Prior ensemble 3+/5 vote voxels")
    lines=ax.get_lines()+ax2.get_lines(); ax.legend(lines,[l.get_label() for l in lines],loc="upper right")
    ax.set_title("LAD plaque transfer after label-blind lumen-frame refinement"); fig.tight_layout(); fig.savefig(out,dpi=170); plt.close(fig)


def _plane_image(geom,src,c,t,half=5.0,step=.15):
    u,v=_orth_basis(t); q=np.arange(-half,half+1e-9,step); yy,xx=np.meshgrid(q,q,indexing="ij")
    P=c+xx[...,None]*u+yy[...,None]*v
    return _sample(geom,src,P.reshape(-1,3)).reshape(len(q),len(q)),q


def _plot_qc(geom,src,centers,smooth_t,stations,out,vessel):
    d=stations[np.isclose(stations.shell_thickness_mm,NOMINAL_SHELL)].copy()
    d["axis_improvement"]=d.raw_lumen_axis_proxy-d.lumen_axis_proxy
    ids=list(d.nlargest(4,"axis_improvement").station_index.astype(int))+list(d.nsmallest(2,"center_refine_offset_mm").station_index.astype(int))
    ids=list(dict.fromkeys(ids))[:6]
    fig,axes=plt.subplots(2,3,figsize=(12,8)); axes=np.asarray(axes).ravel()
    for ax in axes: ax.axis("off")
    for ax,i in zip(axes,ids):
        r=d[d.station_index==i].iloc[0]
        c=np.array([r.refined_center_lps_x_mm,r.refined_center_lps_y_mm,r.refined_center_lps_z_mm])
        im,q=_plane_image(geom,src,c,smooth_t[i])
        ax.imshow(im,cmap="gray",vmin=-100,vmax=800,extent=[q[0],q[-1],q[-1],q[0]])
        rad=float(r.lumen_radius_median_mm); ax.add_patch(plt.Circle((0,0),rad,fill=False,linewidth=1.2))
        ax.add_patch(plt.Circle((0,0),rad+NOMINAL_SHELL,fill=False,linewidth=1.1,linestyle="--"))
        ax.scatter([0],[0],s=8)
        arc=float(r.frozen_arc_mm if vessel=="LAD" else r.arc_mm)
        ax.set_title(f"{arc:.1f} mm | axis {r.raw_lumen_axis_proxy:.2f}→{r.lumen_axis_proxy:.2f} | shift {r.center_refine_offset_mm:.2f}")
        ax.set_xticks([]); ax.set_yticks([]); ax.axis("on")
    fig.suptitle(f"{vessel} automatically selected refined source-CCTA frames")
    fig.tight_layout(); fig.savefig(out,dpi=160); plt.close(fig)


def synthetic_self_test():
    p=np.array([[0.,0.,0.],[1.,0.1,0.],[2.,0.,0.],[3.,-.1,0.],[4.,0.,0.]])
    raw,sm=_smooth_tangents(p,half=2)
    assert raw.shape==sm.shape==(5,3)
    assert np.allclose(np.linalg.norm(sm,axis=1),1.0)
    assert abs(_auc([0,0,1,1],[.1,.2,.8,.9])-1.0)<1e-12
    offs=_candidate_offsets(np.array([1.,0.,0.]))
    assert len(offs)==1+len(CENTER_RING_RADII_MM)*CENTER_RING_DIRECTIONS
    frozen=np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
    ext=np.array([[-2.,0.,0.],[-1.,0.,0.],[0.,0.,0.]])
    joined,elen,gap=_join_frozen_and_extension(frozen,ext)
    assert abs(_arc(joined)[-1]-4.0)<1e-12 and abs(elen-2.0)<1e-12 and gap<1e-12
    return {"ok":True,"candidate_centers":len(offs),"joined_length_mm":float(_arc(joined)[-1])}


def run(drive_root="/content/drive/MyDrive/OpenPlaque",output_dir=None):
    root=Path(drive_root); out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME
    out.mkdir(parents=True,exist_ok=True)
    _write_json(out/"run_state.json",{"status":"STARTED","algorithm":ALGORITHM,"baseline":BASELINE})

    master=_read_json(root/MASTER); bidir=_read_json(root/DISTAL_BIDIR_SUMMARY)
    prev_rca=_read_json(root/PREV_RCA_SUMMARY); prev_lad=_read_json(root/PREV_LAD_SUMMARY)
    if master.get("status")!="CORONARY_ANATOMY_BASELINE_V2_FROZEN": raise RuntimeError("Frozen master prerequisite failed")
    if bidir.get("status")!="LAD_DISTAL_EXTENSION_BIDIRECTIONALLY_SOURCE_CONFIRMED": raise RuntimeError("LAD distal bidirectional prerequisite failed")
    if prev_rca.get("status")!="RCA_SOURCE_SPACE_PLAQUE_EXCESS_SPECIFICITY_PASS": raise RuntimeError("Prior RCA specificity prerequisite failed")
    if prev_lad.get("status")!="LAD_SOURCE_SPACE_PLAQUE_TRANSFER_VALIDATION_FAILED": raise RuntimeError("Expected prior LAD transfer failure not found")

    geom,src,vv=_load_source(root/SOURCE_CACHE)

    rca=_load_path(root/RCA_CENTERLINE)
    rca_st,rca_centers,rca_raw_t,rca_smooth_t=_quantify_path(geom,src,rca,"RCA")
    rca_st.to_csv(out/"RCA_refined_source_space_stations.csv",index=False)
    models=_fit_rca_models(rca_st); models.to_csv(out/"RCA_refined_reference_models.csv",index=False)
    rca_sc=_apply_models(rca_st,models); rca_sc.to_csv(out/"RCA_refined_excess_stations.csv",index=False)
    rca_profiles={s:_profile_1mm(rca_sc,s,"RCA") for s in SHELLS}
    pd.concat(rca_profiles.values(),ignore_index=True).to_csv(out/"RCA_refined_excess_profiles_all_shells.csv",index=False)
    rca_prior=pd.read_csv(_req(root/RCA_PRIOR))
    rca_match,rca_metric=_validation_metrics(rca_profiles[NOMINAL_SHELL],rca_prior,*RCA_HOLDOUT_ARC)
    rca_match.to_csv(out/"RCA_refined_holdout_bins.csv",index=False)
    rca_shell,rca_min_shell=_cross_shell(rca_profiles,0.0,float(_arc(rca)[-1]))
    rca_shell.to_csv(out/"RCA_refined_shell_consistency.csv",index=False)
    rca_nom=rca_sc[np.isclose(rca_sc.shell_thickness_mm,NOMINAL_SHELL)]
    rca_qc=float(rca_nom.station_qc_pass.mean())
    rca_control=bool(rca_qc>=MIN_RCA_QC and _passes_specificity(rca_metric,rca_min_shell))

    frozen=_load_path(root/FROZEN_LAD); extension=_load_path(root/DISTAL_EXTENSION)
    combined,ext_len,join_gap=_join_frozen_and_extension(frozen,extension)
    frozen_len=float(_arc(frozen)[-1])
    lad_st,lad_centers,lad_raw_t,lad_smooth_t=_quantify_path(geom,src,combined,"LAD",frozen_arc_offset=-ext_len)
    lad_st.to_csv(out/"LAD_refined_source_space_stations.csv",index=False)
    lad_sc=_apply_models(lad_st,models); lad_sc.to_csv(out/"LAD_refined_transferred_excess_stations.csv",index=False)
    lad_profiles={s:_profile_1mm(lad_sc,s,"LAD") for s in SHELLS}
    pd.concat(lad_profiles.values(),ignore_index=True).to_csv(out/"LAD_refined_excess_profiles_all_shells.csv",index=False)
    lad_nom=lad_profiles[NOMINAL_SHELL]; lad_nom.to_csv(out/"LAD_refined_excess_profile_nominal.csv",index=False)
    lad_prior=pd.read_csv(_req(root/LAD_PRIOR))
    lad_match,lad_metric=_validation_metrics(lad_nom,lad_prior,0.0,math.floor(frozen_len))
    lad_match.to_csv(out/"LAD_refined_transfer_validation_bins.csv",index=False)
    lad_shell,lad_min_shell=_cross_shell(lad_profiles,0.0,math.floor(frozen_len))
    lad_shell.to_csv(out/"LAD_refined_shell_consistency.csv",index=False)
    lad_nom_st=lad_sc[np.isclose(lad_sc.shell_thickness_mm,NOMINAL_SHELL)]
    frozen_qc=float(lad_nom_st[lad_nom_st.frozen_arc_mm>=-0.25].station_qc_pass.mean())
    ext_qc=float(lad_nom_st[lad_nom_st.frozen_arc_mm<-0.25].station_qc_pass.mean())
    lad_geom=bool(frozen_qc>=MIN_LAD_FROZEN_QC and ext_qc>=MIN_LAD_EXTENSION_QC)
    lad_spec=bool(_passes_specificity(lad_metric,lad_min_shell))

    if not rca_control: status=STATUS_RCA_FAIL
    elif not lad_geom: status=STATUS_LAD_GEOM_FAIL
    elif not lad_spec: status=STATUS_LAD_SPEC_FAIL
    else: status=STATUS_PASS

    diag=[]
    for vessel,d in [("RCA",rca_nom),("LAD",lad_nom_st)]:
        for r in d.itertuples():
            diag.append({
                "vessel":vessel,"station_index":int(r.station_index),
                "arc_mm":float(r.frozen_arc_mm if vessel=="LAD" else r.arc_mm),
                "raw_axis_proxy":float(r.raw_lumen_axis_proxy),"refined_axis_proxy":float(r.lumen_axis_proxy),
                "raw_qc_pass":bool(r.raw_station_qc_pass),"refined_qc_pass":bool(r.station_qc_pass),
                "center_refine_offset_mm":float(r.center_refine_offset_mm),
                "tangent_refine_angle_deg":float(r.tangent_refine_angle_deg)
            })
    pd.DataFrame(diag).to_csv(out/"frame_refinement_diagnostics.csv",index=False)

    _plot_axis_diagnostics(rca_nom,out/"01_RCA_frame_refinement_diagnostics.png","RCA frame refinement")
    _plot_axis_diagnostics(lad_nom_st,out/"02_LAD_frame_refinement_diagnostics.png","LAD frame refinement")
    _plot_transfer(lad_nom,lad_match,ext_len,out/"03_LAD_refined_transfer_vs_prior.png")
    _plot_qc(geom,src,lad_centers,lad_smooth_t,lad_nom_st,out/"04_LAD_refined_source_QC.png","LAD")
    _plot_qc(geom,src,rca_centers,rca_smooth_t,rca_nom,out/"05_RCA_refined_source_QC.png","RCA")

    lad_frozen_prof=lad_nom[(lad_nom.arc_start_mm>=0)&(lad_nom.arc_start_mm<math.ceil(frozen_len))]
    lad_ext_prof=lad_nom[lad_nom.arc_start_mm<0]
    total_col="excess_total_plaque_proxy_mm3"
    summary={
        "status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE,
        "master_status":master.get("status"),"master_modified":False,
        "source_voxel_volume_mm3":vv,
        "prior_rca_specificity_status":prev_rca.get("status"),
        "prior_lad_transfer_status":prev_lad.get("status"),
        "method_change":{
            "label_blind":True,
            "local_tangent":"PCA over +/- 3 resampled 0.5-mm stations",
            "center_search_radius_mm":CENTER_SEARCH_RADIUS_MM,
            "center_candidates":1+len(CENTER_RING_RADII_MM)*CENTER_RING_DIRECTIONS,
            "selection_uses_plaque_labels":False,
            "note":"LAD plaque labels have already been inspected in prior development, so LAD specificity metrics in this experiment are developmental rather than independent external validation."
        },
        "rca_control":{
            "station_qc_fraction":rca_qc,
            "specificity":rca_metric,
            "min_cross_shell_spearman":rca_min_shell,
            "pass":rca_control
        },
        "lad_geometry":{
            "frozen_length_mm":frozen_len,"distal_extension_length_mm":ext_len,
            "combined_length_mm":float(_arc(combined)[-1]),"join_gap_mm":join_gap,
            "frozen_station_qc_fraction":frozen_qc,"extension_station_qc_fraction":ext_qc,
            "geometry_gate_pass":lad_geom
        },
        "lad_developmental_transfer":{
            "specificity":lad_metric,"min_cross_shell_spearman":lad_min_shell,
            "specificity_gate_pass":lad_spec,
            "frozen_excess_total_mm3":float(lad_frozen_prof[total_col].sum()),
            "distal_extension_excess_total_mm3":float(lad_ext_prof[total_col].sum())
        },
        "is_validated_clinical_tpv":False,
        "scientific_boundary":(
            "This experiment changes only source-plane geometry using label-blind local tangent smoothing and constrained center refinement. "
            "The refined RCA 20-50 mm normal-wall model is fit on RCA and transferred unchanged to LAD. "
            "Because prior LAD plaque labels have already been inspected during development, any LAD plaque-label agreement is developmental, "
            "not an untouched external validation. The metric remains a reference-normalized excess-wall proxy, not independently segmented clinical TPV."
        )
    }
    _write_json(out/"summary.json",summary)
    _write_json(out/"input_provenance.json",{
        "source_cache":str(root/SOURCE_CACHE),"rca_centerline":str(root/RCA_CENTERLINE),
        "frozen_lad":str(root/FROZEN_LAD),"distal_extension":str(root/DISTAL_EXTENSION),
        "rca_prior":str(root/RCA_PRIOR),"lad_prior":str(root/LAD_PRIOR),
        "prior_rca_summary":str(root/PREV_RCA_SUMMARY),"prior_lad_summary":str(root/PREV_LAD_SUMMARY)
    })
    report=out/"OPENPLAQUE_CROSS_VESSEL_LUMEN_FRAME_REFINEMENT_PLAQUE_RESCUE_REPORT.html"
    report.write_text(
        "<html><body><h1>OpenPlaque Cross-Vessel Lumen Frame Refinement Plaque Rescue v1</h1>"
        f"<p><b>Status:</b> {status}</p>"
        f"<p>RCA control QC: {rca_qc:.3f}; RCA majority AUC: {rca_metric.get('majority_vs_vote_free_auc',float('nan')):.3f}; "
        f"RCA strict AUC: {rca_metric.get('strict5_vs_vote_free_auc',float('nan')):.3f}; RCA shell rho: {rca_min_shell:.3f}.</p>"
        f"<p>LAD frozen QC: {frozen_qc:.3f}; extension QC: {ext_qc:.3f}; "
        f"LAD majority AUC: {lad_metric.get('majority_vs_vote_free_auc',float('nan')):.3f}; "
        f"LAD strict AUC: {lad_metric.get('strict5_vs_vote_free_auc',float('nan')):.3f}; LAD shell rho: {lad_min_shell:.3f}.</p>"
        "<p><b>Boundary:</b> label-blind geometry refinement, but LAD label agreement is developmental because LAD labels were previously inspected. "
        "Reference-normalized excess-wall proxy only; not clinical TPV.</p>"
        "<h2>RCA control</h2><pre>"+json.dumps(summary["rca_control"],indent=2,default=str)+"</pre>"
        "<h2>LAD geometry</h2><pre>"+json.dumps(summary["lad_geometry"],indent=2,default=str)+"</pre>"
        "<h2>LAD developmental transfer</h2><pre>"+json.dumps(summary["lad_developmental_transfer"],indent=2,default=str)+"</pre>"
        '<img src="01_RCA_frame_refinement_diagnostics.png" style="max-width:100%">'
        '<img src="02_LAD_frame_refinement_diagnostics.png" style="max-width:100%">'
        '<img src="03_LAD_refined_transfer_vs_prior.png" style="max-width:100%">'
        '<img src="04_LAD_refined_source_QC.png" style="max-width:100%">'
        '<img src="05_RCA_refined_source_QC.png" style="max-width:100%">'
        "</body></html>",encoding="utf-8"
    )
    _write_json(out/"run_state.json",{"status":"COMPLETE","result_status":status,"algorithm":ALGORITHM,"baseline":BASELINE})
    zpath=out/"OPENPLAQUE_CROSS_VESSEL_LUMEN_FRAME_REFINEMENT_PLAQUE_RESCUE_RESULTS.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p!=zpath:z.write(p,p.name)
    return {"summary":summary,"report":str(report),"zip":str(zpath)}
