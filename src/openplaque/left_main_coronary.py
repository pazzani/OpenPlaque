from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.filters import frangi
from skimage.graph import MCP_Geometric

ALGORITHM_VERSION = "left-main-bifurcation-v1.0"


def arc_mm(path_zyx, spacing_zyx):
    p = np.asarray(path_zyx, float)
    if len(p) < 2:
        return np.array([0.0])
    d = np.diff(p, axis=0) * np.asarray(spacing_zyx)[None, :]
    return np.r_[0.0, np.cumsum(np.sqrt((d * d).sum(axis=1)))]


def resample_path(path_zyx, spacing_zyx, step_mm=0.6):
    p = np.asarray(path_zyx, float)
    s = arc_mm(p, spacing_zyx)
    keep = np.r_[True, np.diff(s) > 1e-6]
    p, s = p[keep], s[keep]
    if len(p) < 2 or s[-1] < step_mm:
        return p
    su = np.arange(0.0, s[-1] + 1e-6, step_mm)
    return np.column_stack([np.interp(su, s, p[:, j]) for j in range(3)])


def source_to_ds(path_source_zyx, lo_zyx, zoom_zyx):
    return (np.asarray(path_source_zyx, float) - np.asarray(lo_zyx, float)) * np.asarray(zoom_zyx, float)


def ds_to_source(path_ds_zyx, lo_zyx, zoom_zyx):
    return np.asarray(lo_zyx, float) + np.asarray(path_ds_zyx, float) / np.asarray(zoom_zyx, float)


def physical_lps_from_zyx(image, zyx):
    return np.asarray([
        image.TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z)))
        for z, y, x in np.asarray(zyx, float)
    ], float)


def build_downsampled_evidence(ct, aorta, spacing_zyx, seed_source_zyx,
                               half_mm=(82, 96, 96), target_mm=1.0):
    ct = np.asarray(ct, np.float32)
    aorta = np.asarray(aorta, bool)
    spacing_zyx = np.asarray(spacing_zyx, float)
    seed = np.asarray(seed_source_zyx, float)
    half_vox = np.ceil(np.asarray(half_mm, float) / spacing_zyx).astype(int)
    o = np.rint(seed).astype(int)
    lo = np.maximum(0, o - half_vox)
    hi = np.minimum(np.asarray(ct.shape), o + half_vox + 1)
    crop = ct[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    acrop = aorta[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    zoom = np.minimum(1.0, spacing_zyx / float(target_mm))
    ds_spacing = spacing_zyx / zoom
    dct = ndi.zoom(crop, zoom=zoom, order=1, mode="nearest", prefilter=False).astype(np.float32)
    daorta = ndi.zoom(acrop.astype(np.uint8), zoom=zoom, order=0,
                      mode="nearest", prefilter=False) > 0

    sm = gaussian_filter(dct, sigma=np.maximum(0.55 / ds_spacing, 0.45))
    intensity = np.clip((sm - 140.0) / 560.0, 0.0, 1.0)
    intensity[(sm < 120) | (sm > 1000)] = 0
    norm = np.clip((sm + 100.0) / 1050.0, 0, 1)
    vesselness = np.nan_to_num(frangi(norm, sigmas=(0.7, 1.0, 1.4, 1.8, 2.2),
                                      black_ridges=False))
    positive = vesselness[vesselness > 0]
    v99 = np.percentile(positive, 99.2) if positive.size else 1.0
    vesselness = np.clip(vesselness / max(v99, 1e-8), 0, 1).astype(np.float32)
    small = gaussian_filter(sm, sigma=np.maximum(0.8 / ds_spacing, 0.5))
    broad = gaussian_filter(sm, sigma=np.maximum(4.0 / ds_spacing, 1.5))
    dog = np.maximum(small - broad, 0)
    p99 = np.percentile(dog[np.isfinite(dog)], 99.2) if np.any(np.isfinite(dog)) else 1.0
    dog = np.clip(dog / max(p99, 1e-6), 0, 1).astype(np.float32)
    support = (0.50 * intensity + 0.38 * vesselness + 0.12 * dog).astype(np.float32)

    bright = sm >= 150
    lumen_r = ndi.distance_transform_edt(bright, sampling=ds_spacing).astype(np.float32)
    dist_aorta = ndi.distance_transform_edt(~daorta, sampling=ds_spacing).astype(np.float32)
    large = np.clip((lumen_r - 3.8) / 3.0, 0, 1).astype(np.float32)
    cost = 1.0 / (0.04 + support) + 7.0 * large
    cost[sm < 120] += 16
    cost[daorta] = 1e5
    cost[(dist_aorta < 1.3) & (~daorta)] += 30
    return {
        "ct": dct, "aorta": daorta, "support": support,
        "vesselness": vesselness, "dog": dog,
        "lumen_radius_mm": lumen_r, "dist_aorta_mm": dist_aorta,
        "cost": cost.astype(np.float32),
        "lo_source_zyx": lo.astype(int), "hi_source_zyx": hi.astype(int),
        "zoom_zyx": zoom.astype(float), "spacing_zyx": ds_spacing.astype(float),
    }


def _aorta_center_at_z(aorta, z):
    z = int(np.clip(round(z), 0, aorta.shape[0] - 1))
    for dz in range(0, 8):
        for zz in {z-dz, z+dz}:
            if 0 <= zz < aorta.shape[0]:
                yy, xx = np.where(aorta[zz])
                if len(yy) >= 20:
                    return np.array([float(zz), float(np.mean(yy)), float(np.mean(xx))])
    raise RuntimeError("Could not estimate local aortic center")


def detect_left_ostia(evd, rca_seed_ds, topn=18):
    """Return multiple left-coronary ostium candidates; no candidate is accepted yet."""
    support, ct, aorta = evd["support"], evd["ct"], evd["aorta"]
    da, lr, sp = evd["dist_aorta_mm"], evd["lumen_radius_mm"], evd["spacing_zyx"]
    rca_seed_ds = np.asarray(rca_seed_ds, float)
    ctr = _aorta_center_at_z(aorta, rca_seed_ds[0])
    rca_vec = (rca_seed_ds - ctr) * sp
    rca_vec[0] *= 0.35
    rca_vec /= max(np.linalg.norm(rca_vec), 1e-6)
    zz, yy, xx = np.indices(aorta.shape)
    dz_mm = np.abs((zz - rca_seed_ds[0]) * sp[0])
    shell = ((~aorta) & (da >= 1.2) & (da <= 5.8) & (dz_mm <= 15.0) &
             (ct >= 150) & (ct <= 950) & (lr <= 4.8))
    if np.any(shell):
        shell &= support >= np.percentile(support[shell], 65)
    coords = np.argwhere(shell)
    rows = []
    for p in coords:
        c = _aorta_center_at_z(aorta, p[0])
        rv = (p.astype(float) - c) * sp
        rv[0] *= 0.35
        n = np.linalg.norm(rv)
        if n < 2:
            continue
        u = rv / n
        opposite = -float(np.dot(u, rca_vec))
        if opposite < -0.05:
            continue
        rs, rh = [], []
        for mm in np.linspace(2, 14, 9):
            q = (p * sp + u * mm) / sp
            if np.any(q < 1) or np.any(q >= np.asarray(ct.shape)-2):
                continue
            rs.append(float(map_coordinates(support, q[:,None], order=1, mode="nearest")[0]))
            rh.append(float(map_coordinates(ct, q[:,None], order=1, mode="nearest")[0]))
        if len(rs) < 6:
            continue
        rs, rh = np.asarray(rs), np.asarray(rh)
        plausible = float(np.mean((rh >= 120) & (rh <= 950)))
        score = (0.44*float(np.mean(rs)) + 0.22*float(np.percentile(rs,25)) +
                 0.18*opposite + 0.11*plausible + 0.05*float(support[tuple(p)]))
        rows.append({"score":score, "z":float(p[0]), "y":float(p[1]), "x":float(p[2]),
                     "dir_z":float(u[0]), "dir_y":float(u[1]), "dir_x":float(u[2]),
                     "opposite_rca":opposite, "outward_mean_support":float(np.mean(rs)),
                     "outward_plausible_hu":plausible})
    if not rows:
        raise RuntimeError("No viable left ostium candidates")
    df = pd.DataFrame(rows).sort_values("score", ascending=False).head(topn).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df)+1))
    return df


def _mcp_paths(cost, start, endpoints):
    mcp = MCP_Geometric(np.asarray(cost, float), fully_connected=True)
    mcp.find_costs([tuple(np.rint(start).astype(int))])
    out = []
    for ep in endpoints:
        try:
            p = np.asarray(mcp.traceback(tuple(np.rint(ep).astype(int))), int)
        except Exception:
            continue
        if len(p) >= 2:
            out.append(p)
    return out


def _path_metrics(path, evd, heading=None):
    p = np.asarray(path, int)
    sp = evd["spacing_zyx"]
    s = arc_mm(p, sp); L = float(s[-1])
    su = evd["support"][tuple(p.T)]; hu = evd["ct"][tuple(p.T)]
    da = evd["dist_aorta_mm"][tuple(p.T)]; lr = evd["lumen_radius_mm"][tuple(p.T)]
    step = np.diff(p.astype(float), axis=0) * sp
    if len(step) >= 2:
        u = step / np.maximum(np.linalg.norm(step, axis=1, keepdims=True), 1e-6)
        ang = np.arccos(np.clip(np.sum(u[:-1]*u[1:], axis=1), -1, 1))
        mean_turn, p95_turn = float(np.mean(ang)), float(np.percentile(ang,95))
    else:
        mean_turn = p95_turn = math.pi
    heading_angle = 0.0
    if heading is not None and len(p) > 4:
        first = (p[min(len(p)-1,6)] - p[0]) * sp
        heading_angle = float(np.degrees(np.arccos(np.clip(
            np.dot(first, heading)/(np.linalg.norm(first)*max(np.linalg.norm(heading),1e-6)+1e-9), -1, 1))))
    after = s >= min(3.0, 0.25*L)
    return {"length_mm":L, "mean_support":float(np.mean(su)),
            "p10_support":float(np.percentile(su,10)), "mean_hu":float(np.mean(hu)),
            "p10_hu":float(np.percentile(hu,10)),
            "aorta_reentry_fraction":float(np.mean(da[after] < 1.3)) if np.any(after) else float(np.mean(da<1.3)),
            "large_lumen_fraction":float(np.mean(lr > 5.0)),
            "mean_lumen_radius_mm":float(np.mean(lr)), "end_dist_aorta_mm":float(da[-1]),
            "mean_turn_rad":mean_turn, "p95_turn_rad":p95_turn,
            "heading_angle_deg":heading_angle}


def _endpoint_candidates(evd, start, heading, min_r, max_r, min_forward,
                         max_endpoints=40, support_percentile=78):
    support, da, lr, ct, sp = (evd["support"], evd["dist_aorta_mm"], evd["lumen_radius_mm"],
                                evd["ct"], evd["spacing_zyx"])
    G = np.indices(support.shape, dtype=np.float32).reshape(3,-1).T
    D = (G - np.asarray(start)[None,:]) * sp[None,:]
    rad = np.linalg.norm(D, axis=1)
    h = np.asarray(heading, float); h /= max(np.linalg.norm(h), 1e-6)
    forward = D @ h
    su, dd, rr, hv = support.ravel(), da.ravel(), lr.ravel(), ct.ravel()
    positive = support[support > 0]
    sth = np.percentile(positive, support_percentile) if positive.size else 0.2
    m = ((rad >= min_r) & (rad <= max_r) & (forward >= min_forward) &
         (su >= sth) & (dd >= 1.5) & (rr <= 5.0) & (hv >= 120) & (hv <= 1000))
    cand = G[m].astype(int)
    if len(cand) == 0:
        return []
    val = su[m] + 0.025*np.minimum(dd[m],16) - 0.08*np.maximum(rr[m]-3.5,0) + 0.002*np.minimum(forward[m],80)
    order = np.argsort(val)[::-1]; eps = []
    for j in order:
        p = cand[j]
        if all(np.linalg.norm((p-q)*sp) >= 4.0 for q in eps):
            eps.append(p)
        if len(eps) >= max_endpoints:
            break
    return eps


def trace_routes(evd, start_ds, heading_mm, min_len, max_len, max_routes=30,
                 min_forward=3.0, support_percentile=78):
    eps = _endpoint_candidates(evd, start_ds, heading_mm, min_len, max_len,
                               min_forward, max_endpoints=max(40,max_routes*2),
                               support_percentile=support_percentile)
    if not eps:
        return []
    routes = []
    for p in _mcp_paths(evd["cost"], start_ds, eps):
        m = _path_metrics(p, evd, heading_mm)
        if not (min_len <= m["length_mm"] <= max_len):
            continue
        if m["aorta_reentry_fraction"] > 0.04 or m["large_lumen_fraction"] > 0.18:
            continue
        score = (1.9*m["mean_support"] + 0.65*m["p10_support"] +
                 0.010*min(m["length_mm"],90) + 0.020*min(m["end_dist_aorta_mm"],18) -
                 0.34*m["mean_turn_rad"] - 0.11*m["p95_turn_rad"] -
                 0.005*m["heading_angle_deg"] - 1.1*m["large_lumen_fraction"])
        routes.append({"path":p, "graph_score":float(score), **m})
    routes.sort(key=lambda r:r["graph_score"], reverse=True)
    return routes[:max_routes]


def _orthogonal_basis(t):
    t = np.asarray(t,float); t /= max(np.linalg.norm(t),1e-8)
    ref = np.array([1.,0.,0.]) if abs(t[0]) < 0.82 else np.array([0.,1.,0.])
    u = np.cross(t, ref); u /= max(np.linalg.norm(u),1e-8)
    v = np.cross(t, u); v /= max(np.linalg.norm(v),1e-8)
    return u, v


def orthogonal_plane(ct, point_zyx, tangent_zyx_mm, spacing_zyx, half_mm=6.0, pix_mm=0.18):
    u, v = _orthogonal_basis(tangent_zyx_mm)
    c = np.arange(-half_mm, half_mm+1e-9, pix_mm)
    U, V = np.meshgrid(c,c,indexing="xy")
    pm = np.asarray(point_zyx,float)*np.asarray(spacing_zyx,float)
    pos = pm[None,None,:] + U[...,None]*u[None,None,:] + V[...,None]*v[None,None,:]
    vox = pos/np.asarray(spacing_zyx,float)[None,None,:]
    im = map_coordinates(np.asarray(ct,float), [vox[...,0],vox[...,1],vox[...,2]], order=1, mode="nearest")
    return im, c


def plane_lumen_metrics(im, coords_mm):
    im = np.asarray(im,float); c = np.asarray(coords_mm,float)
    h,w = im.shape; cy,cx = h//2,w//2
    yy,xx = np.mgrid[:h,:w]
    Y,X = c[yy],c[xx]
    R = np.sqrt(X*X+Y*Y)
    center_hu = float(np.median(im[R <= 0.8]))
    threshold = float(np.clip(0.55*center_hu, 180, 520))
    mask = (im >= threshold) & (im <= 1200) & (R <= 5.5)
    lab,n = ndi.label(mask)
    central_labels = lab[R <= 0.55]
    central_labels = central_labels[central_labels > 0]
    if central_labels.size == 0:
        return {"center_hu":center_hu,"threshold_hu":threshold,"radius_mm":np.nan,
                "centroid_offset_mm":np.inf,"circularity":0.0,"core_minus_ring_hu":np.nan,
                "component_area_mm2":0.0,"plane_score":0.0,"plane_pass":False}
    vals,counts = np.unique(central_labels,return_counts=True)
    k = int(vals[np.argmax(counts)]); comp = lab==k
    pix = float(abs(c[1]-c[0])) if len(c)>1 else 0.18
    area = float(comp.sum()*pix*pix); radius = math.sqrt(area/math.pi)
    weights = comp.astype(float)
    cy_mm = float((Y*weights).sum()/max(weights.sum(),1)); cx_mm = float((X*weights).sum()/max(weights.sum(),1))
    offset = float(math.hypot(cx_mm,cy_mm))
    edge = comp & ~ndi.binary_erosion(comp)
    perimeter = float(edge.sum()*pix)
    circularity = float(np.clip(4*math.pi*area/max(perimeter*perimeter,1e-6),0,1.2))
    core = float(np.mean(im[R<=1.0]))
    ring_mask = (R>=3.0)&(R<=5.0)
    ring = float(np.mean(im[ring_mask])) if np.any(ring_mask) else core
    contrast = core-ring
    return {"center_hu":center_hu,"threshold_hu":threshold,"radius_mm":radius,
            "centroid_offset_mm":offset,"circularity":circularity,
            "core_minus_ring_hu":contrast,"component_area_mm2":area}


def serial_lumen_qc(path_source_zyx, ct, spacing_zyx, rca_calibration=None,
                    n_samples=10, label_name="candidate"):
    p = resample_path(path_source_zyx, spacing_zyx, 0.45)
    s = arc_mm(p, spacing_zyx)
    if len(p)<5 or s[-1] < 2:
        return pd.DataFrame(), {"label":label_name,"length_mm":float(s[-1]) if len(s) else 0,
                                "median_plane_score":0.0,"plane_pass_fraction":0.0}
    sample_s = np.linspace(min(1.2,s[-1]*0.08), max(min(1.2,s[-1]*0.08),s[-1]-1.2), n_samples)
    rows=[]
    rca_r = float(rca_calibration.get("median_radius_mm",1.8)) if rca_calibration else 1.8
    rca_h = float(rca_calibration.get("median_center_hu",550)) if rca_calibration else 550
    for ss in sample_s:
        i=int(np.argmin(np.abs(s-ss))); i0=max(0,i-3); i1=min(len(p)-1,i+3)
        t=(p[i1]-p[i0])*np.asarray(spacing_zyx,float)
        if np.linalg.norm(t)<1e-6: continue
        im,c=orthogonal_plane(ct,p[i],t,spacing_zyx)
        m=plane_lumen_metrics(im,c)
        r=m["radius_mm"]
        if np.isfinite(r):
            radius_score=float(np.exp(-0.5*((r/(1.25*rca_r+1e-6)-1)/0.60)**2))
        else: radius_score=0.0
        offset_score=float(np.exp(-0.5*(m["centroid_offset_mm"]/0.85)**2)) if np.isfinite(m["centroid_offset_mm"]) else 0.0
        circ_score=float(np.clip(m["circularity"]/0.55,0,1))
        contrast_score=float(1/(1+np.exp(-(m["core_minus_ring_hu"]-20)/90))) if np.isfinite(m["core_minus_ring_hu"]) else 0.0
        hu_score=float(np.exp(-0.5*((m["center_hu"]-rca_h)/330)**2))
        score=0.30*radius_score+0.27*offset_score+0.20*circ_score+0.14*contrast_score+0.09*hu_score
        pass_plane=(np.isfinite(r) and 0.45*rca_r <= r <= min(5.0,2.3*rca_r+0.4) and
                    m["centroid_offset_mm"] <= 1.45 and m["circularity"] >= 0.23 and
                    120 <= m["center_hu"] <= 1100)
        rows.append({"label":label_name,"arc_mm":float(s[i]),**m,"plane_score":float(score),"plane_pass":bool(pass_plane)})
    df=pd.DataFrame(rows)
    summary={"label":label_name,"length_mm":float(s[-1]),
             "median_plane_score":float(df.plane_score.median()) if len(df) else 0.0,
             "plane_pass_fraction":float(df.plane_pass.mean()) if len(df) else 0.0,
             "median_radius_mm":float(df.radius_mm.median()) if len(df) else np.nan,
             "median_offset_mm":float(df.centroid_offset_mm.median()) if len(df) else np.nan,
             "median_circularity":float(df.circularity.median()) if len(df) else np.nan,
             "median_center_hu":float(df.center_hu.median()) if len(df) else np.nan,
             "median_core_minus_ring_hu":float(df.core_minus_ring_hu.median()) if len(df) else np.nan}
    return df,summary


def calibrate_from_rca(rca_path_source, ct, spacing_zyx):
    df,summary=serial_lumen_qc(rca_path_source,ct,spacing_zyx,rca_calibration=None,n_samples=12,label_name="RCA_reference")
    return df,summary


def score_left_main_candidates(evd, ostia_df, ct_source, spacing_source, rca_calibration,
                               lo_source_zyx, zoom_zyx, max_ostia=10):
    candidates=[]; rows=[]
    for _,o in ostia_df.head(max_ostia).iterrows():
        seed=np.array([o.z,o.y,o.x],float); heading=np.array([o.dir_z,o.dir_y,o.dir_x],float)
        routes=trace_routes(evd,seed,heading,min_len=6,max_len=30,max_routes=14,min_forward=2.5,support_percentile=72)
        for r in routes:
            src=ds_to_source(r["path"],lo_source_zyx,zoom_zyx)
            qcdf,qcs=serial_lumen_qc(src,ct_source,spacing_source,rca_calibration,n_samples=9,label_name="left_main")
            length_quality=float(np.exp(-0.5*((r["length_mm"]-14)/8.0)**2))
            score=(0.42*r["graph_score"]+0.34*qcs["median_plane_score"]+
                   0.19*qcs["plane_pass_fraction"]+0.05*length_quality)
            rec={"ostium_rank":int(o["rank"]),"combined_score":float(score),
                 **{k:v for k,v in r.items() if k!="path"},**{f"serial_{k}":v for k,v in qcs.items() if k!="label"}}
            rows.append(rec); candidates.append({"path_ds":r["path"],"path_source":src,"qc_df":qcdf,"summary":rec})
    if not candidates: raise RuntimeError("No left-main route candidates")
    order=np.argsort([c["summary"]["combined_score"] for c in candidates])[::-1]
    candidates=[candidates[i] for i in order]
    table=pd.DataFrame(rows).sort_values("combined_score",ascending=False).reset_index(drop=True)
    table.insert(0,"rank",np.arange(1,len(table)+1))
    return candidates,table


def _common_length_mm(a,b,sp,tol_mm=3.0):
    aa=resample_path(a,sp,1.0)*np.asarray(sp)[None,:]
    bb=resample_path(b,sp,1.0)*np.asarray(sp)[None,:]
    n=min(len(aa),len(bb))
    if n<2:return 0.0
    d=np.linalg.norm(aa[:n]-bb[:n],axis=1)
    idx=np.where(d>tol_mm)[0]
    return float(idx[0]) if len(idx) else float(n-1)


def select_branch_pair_from_trunk(evd, trunk_source, ct_source, spacing_source,
                                  rca_calibration, image, lo_source_zyx, zoom_zyx,
                                  max_routes=22):
    p=resample_path(trunk_source,spacing_source,0.7)
    s=arc_mm(p,spacing_source)
    i0=max(0,len(p)-8); heading=(p[-1]-p[i0])*np.asarray(spacing_source)
    heading/=max(np.linalg.norm(heading),1e-6)
    seed_ds=source_to_ds(p[-1],lo_source_zyx,zoom_zyx)
    routes=trace_routes(evd,seed_ds,heading,min_len=28,max_len=135,max_routes=max_routes,
                        min_forward=6,support_percentile=78)
    enriched=[]
    for r in routes:
        src=ds_to_source(r["path"],lo_source_zyx,zoom_zyx)
        qcdf,qcs=serial_lumen_qc(src,ct_source,spacing_source,rca_calibration,n_samples=11,label_name="branch")
        r2=dict(r); r2["path_source"]=src; r2["qc_df"]=qcdf; r2["qc_summary"]=qcs
        r2["branch_score"]=float(0.55*r["graph_score"]+0.27*qcs["median_plane_score"]+0.18*qcs["plane_pass_fraction"])
        enriched.append(r2)
    enriched.sort(key=lambda x:x["branch_score"],reverse=True)
    pairs=[]
    for i in range(len(enriched)):
        for j in range(i+1,len(enriched)):
            a,b=enriched[i],enriched[j]
            common=_common_length_mm(a["path"],b["path"],evd["spacing_zyx"],tol_mm=3.0)
            sep=float(np.linalg.norm((a["path"][-1]-b["path"][-1])*evd["spacing_zyx"]))
            if common>18 or sep<24: continue
            if min(a["qc_summary"]["plane_pass_fraction"],b["qc_summary"]["plane_pass_fraction"])<0.40: continue
            divergence_quality=float(np.exp(-0.5*((common-5)/6.5)**2))
            score=a["branch_score"]+b["branch_score"]+0.012*min(sep,80)+0.20*divergence_quality
            pairs.append((score,common,sep,a,b))
    if not pairs:
        return None,enriched
    pairs.sort(key=lambda x:x[0],reverse=True)
    score,common,sep,a,b=pairs[0]
    start_ph=physical_lps_from_zyx(image,[trunk_source[-1]])[0]
    pha=physical_lps_from_zyx(image,[a["path_source"][-1]])[0]
    phb=physical_lps_from_zyx(image,[b["path_source"][-1]])[0]
    da,db=pha-start_ph,phb-start_ph
    lad_a=-1.00*da[2]-0.45*da[1]+0.10*da[0]
    lad_b=-1.00*db[2]-0.45*db[1]+0.10*db[0]
    lad,lcx=(a,b) if lad_a>=lad_b else (b,a)
    return {"pair_score":float(score),"common_after_trunk_mm":float(common),
            "endpoint_separation_mm":float(sep),"LAD":lad,"LCX":lcx,
            "lad_metric_a":float(lad_a),"lad_metric_b":float(lad_b)},enriched
