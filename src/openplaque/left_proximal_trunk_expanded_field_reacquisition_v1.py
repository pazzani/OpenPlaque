from __future__ import annotations

"""Expanded-field source-CCTA reacquisition from the dense-QC accepted proximal-trunk endpoint.

Prerequisite:
  Left_Proximal_Trunk_Continuation_QC_v1 must be PROXIMAL_TRUNK_CONTINUATION_QC_POSITIVE.

Scientific question:
  The earlier 46-mm LAD->aorta beam began rejecting large numbers of proposals as outside its
  narrow source field near the same arc where dense plane QC later validated a 20-mm proximal
  continuation. This experiment starts at that independently QC-accepted endpoint and expands
  only the source search field margin. The source HU, vesselness, coronary-mask proximity,
  turning, monotonic aortic-distance, step size, and beam-width rules remain unchanged.

A positive aortic bridge is only a source-supported proximal-trunk-to-aorta candidate for visual
adjudication. It is not labeled LM, does not establish LCX topology, and does not modify the
frozen master anatomy.
"""

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
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "left-proximal-trunk-expanded-field-reacquisition-v1.0"
OUTPUT_DIRNAME = "Left_Proximal_Trunk_Expanded_Field_Reacquisition_v1"
PRIOR_DIRNAME = "Left_Proximal_Trunk_Continuation_QC_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
RCA_PATH = Path("PCAT_RCA_10_50/rca_centerline_smoothed_zyx.csv")
TS = Path("TotalSegmentator_Cardiovascular_Cache_v1")
CUR = TS / "coronary_arteries/coronary_arteries.nii.gz"
LEG = TS / "coronary_arteries_LEGACY/coronary_arteries.nii.gz"
AORTA = TS / "heartchambers_highres/aorta.nii.gz"

PRIOR_SUMMARY = Path(PRIOR_DIRNAME) / "summary.json"
PRIOR_ACCEPTED_PATH = Path(PRIOR_DIRNAME) / "accepted_proximal_trunk_continuation_candidate.csv"

FIELD_MARGIN_MM = 18.0
SEARCH_MAX_MM = 40.0
GATE_MAX_MM = 38.0
PROSPECTIVE_MARGIN_MM = 8.0
STEP_MM = 0.40
BEAM_WIDTH = 110
PLANE_STEP_MM = 0.40
SUSTAINED_FAIL_N = 3
MIN_EXTENSION_MM = 5.0
MIN_PLANE_PASS_FRACTION = 0.80
RCA_ENDPOINT_SEPARATION_MM = 8.0

STATUS_RCA_FAIL = "EXPANDED_FIELD_RCA_PLANE_CONTROL_FAILED"
STATUS_BRIDGE = "PROXIMAL_TRUNK_EXPANDED_FIELD_REACHES_AORTA_REQUIRES_VISUAL_QC"
STATUS_RCA_ASSOC = "PROXIMAL_TRUNK_EXPANDED_FIELD_AORTIC_ENDPOINT_RCA_ASSOCIATED"
STATUS_EXTEND = "PROXIMAL_TRUNK_EXPANDED_FIELD_CONTINUATION_QC_POSITIVE_NO_AORTA"
STATUS_FAIL = "PROXIMAL_TRUNK_EXPANDED_FIELD_NO_VALID_EXTENSION"


def _req(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")


def _source(cache):
    arr = np.load(_req(cache / "series7_int16.npy"), mmap_mode="r")
    meta = json.loads(_req(cache / "series7_int16.json").read_text())
    ref = sitk.GetImageFromArray(np.asarray(arr))
    sp = np.asarray(meta["spacing_zyx"], float)
    ref.SetSpacing(tuple(sp[::-1]))
    ref.SetOrigin(tuple(np.asarray(meta["positions_lps_mm"][0], float)))
    iop = np.asarray(meta["image_orientation_patient"], float)
    row, col = iop[:3], iop[3:]
    slc = np.cross(row, col)
    D = np.array([[row[0], col[0], slc[0]],
                  [row[1], col[1], slc[1]],
                  [row[2], col[2], slc[2]]], float)
    ref.SetDirection(tuple(D.ravel()))
    return ref, arr, sp


def _xyz_to_zyx(img, p):
    p = np.atleast_2d(np.asarray(p, float))
    o = np.asarray(img.GetOrigin())
    sp = np.asarray(img.GetSpacing())
    D = np.asarray(img.GetDirection()).reshape(3, 3)
    return (((p - o) @ np.linalg.inv(D).T) / sp)[:, ::-1]


def _zyx_to_xyz(img, p):
    p = np.atleast_2d(np.asarray(p, float))
    q = p[:, ::-1]
    o = np.asarray(img.GetOrigin())
    sp = np.asarray(img.GetSpacing())
    D = np.asarray(img.GetDirection()).reshape(3, 3)
    return o + (q * sp) @ D.T


def _sample(img, a, p, cval=-1024.0):
    return map_coordinates(np.asarray(a), _xyz_to_zyx(img, p).T,
                           order=1, mode="constant", cval=cval)


def _load_path(path, ref):
    d = pd.read_csv(_req(path))
    for c in (("lps_x_mm", "lps_y_mm", "lps_z_mm"),
              ("x_mm", "y_mm", "z_mm")):
        if all(x in d.columns for x in c):
            return d[list(c)].to_numpy(float)
    for c in (("zyx_z", "zyx_y", "zyx_x"),
              ("source_z", "source_y", "source_x"),
              ("z", "y", "x")):
        if all(x in d.columns for x in c):
            return _zyx_to_xyz(ref, d[list(c)].to_numpy(float))
    raise ValueError(f"No recognized coordinate columns in {path}: {list(d.columns)}")


def _arc(p):
    p = np.asarray(p, float)
    if len(p) <= 1:
        return np.zeros(len(p))
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def _resample(p, step=0.20):
    p = np.asarray(p, float)
    a = _arc(p)
    if len(p) < 2 or a[-1] <= 0:
        return p.copy(), a
    q = np.arange(0, a[-1] + 1e-9, step)
    if q[-1] < a[-1] - 1e-6:
        q = np.r_[q, a[-1]]
    return np.column_stack([np.interp(q, a, p[:, k]) for k in range(3)]), q


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


def _orth_basis(t):
    t = _unit(t)
    axes = np.eye(3)
    seed = axes[np.argmin(np.abs(axes @ t))]
    u = _unit(np.cross(t, seed))
    v = _unit(np.cross(t, u))
    return u, v


def _cone_dirs(t):
    t = _unit(t)
    u, v = _orth_basis(t)
    out = [t]
    for deg in (10, 20, 30, 40, 50, 60):
        a = np.deg2rad(deg)
        for phi in np.linspace(0, 2*np.pi, 12, endpoint=False):
            out.append(_unit(np.cos(a)*t + np.sin(a)*(np.cos(phi)*u + np.sin(phi)*v)))
    return out


def _resample_mask(path, ref):
    im = sitk.ReadImage(str(_req(path)))
    same = (im.GetSize() == ref.GetSize()
            and np.allclose(im.GetSpacing(), ref.GetSpacing())
            and np.allclose(im.GetOrigin(), ref.GetOrigin())
            and np.allclose(im.GetDirection(), ref.GetDirection()))
    if not same:
        im = sitk.Resample(im, ref, sitk.Transform(),
                           sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(im) > 0


def _surface_tree(mask, ref, stride=2):
    surf = mask & ~ndi.binary_erosion(mask, iterations=1, border_value=0)
    z = np.argwhere(surf)[::stride]
    pts = _zyx_to_xyz(ref, z.astype(float))
    return cKDTree(pts.astype(np.float32)), pts


def _orient_endpoint_nearest_tree(path, tree):
    p = np.asarray(path, float)
    d0 = float(tree.query(p[0])[0])
    d1 = float(tree.query(p[-1])[0])
    if d0 <= d1:
        return p.copy(), d0, False
    return p[::-1].copy(), d1, True


def _frangi_3d(vol, spacing, scales=(0.55, 0.80, 1.10, 1.45)):
    x = np.clip(np.asarray(vol, np.float32), 80, 1000)
    x = (x - 80) / 920.0
    spacing = np.asarray(spacing, float)
    best = np.zeros_like(x, np.float32)
    for si, sm in enumerate(scales, start=1):
        print(f"[expanded-field] vesselness scale {si}/{len(scales)} = {sm:.2f} mm")
        sig = np.maximum(sm / spacing, 0.55)
        n = sm * sm
        hzz = ndi.gaussian_filter(x, sig, order=(2,0,0), mode="nearest") * n / (spacing[0]**2)
        hyy = ndi.gaussian_filter(x, sig, order=(0,2,0), mode="nearest") * n / (spacing[1]**2)
        hxx = ndi.gaussian_filter(x, sig, order=(0,0,2), mode="nearest") * n / (spacing[2]**2)
        hzy = ndi.gaussian_filter(x, sig, order=(1,1,0), mode="nearest") * n / (spacing[0]*spacing[1])
        hzx = ndi.gaussian_filter(x, sig, order=(1,0,1), mode="nearest") * n / (spacing[0]*spacing[2])
        hyx = ndi.gaussian_filter(x, sig, order=(0,1,1), mode="nearest") * n / (spacing[1]*spacing[2])
        H = np.empty(x.shape + (3,3), np.float32)
        H[...,0,0] = hzz; H[...,1,1] = hyy; H[...,2,2] = hxx
        H[...,0,1] = H[...,1,0] = hzy
        H[...,0,2] = H[...,2,0] = hzx
        H[...,1,2] = H[...,2,1] = hyx
        vals = np.linalg.eigvalsh(H)
        vals = np.take_along_axis(vals, np.argsort(np.abs(vals), axis=-1), axis=-1)
        l1, l2, l3 = vals[...,0], vals[...,1], vals[...,2]
        eps = 1e-8
        ra = np.abs(l2)/(np.abs(l3)+eps)
        rb = np.abs(l1)/np.sqrt(np.abs(l2*l3)+eps)
        ss = np.sqrt(l1*l1 + l2*l2 + l3*l3)
        nz = ss[ss > 0]
        c = max(float(np.percentile(nz, 90))*0.45 if nz.size else 0.05, 1e-4)
        v = (1-np.exp(-(ra*ra)/0.5))*np.exp(-(rb*rb)/0.5)*(1-np.exp(-(ss*ss)/(2*c*c)))
        v[(l2 >= 0) | (l3 >= 0)] = 0
        best = np.maximum(best, np.nan_to_num(v).astype(np.float32))
    return best


def _expanded_field(ref, src, spacing, cur, leg, start, goal, accepted_path,
                    margin_mm=FIELD_MARGIN_MM):
    ap, aq = _resample(accepted_path, 0.35)
    known = ap[aq >= max(0.0, aq[-1] - 6.0)]
    pts = np.vstack([start[None, :], goal[None, :], known])
    z = _xyz_to_zyx(ref, pts)
    margin_vox = np.asarray([margin_mm, margin_mm, margin_mm]) / spacing
    lo = np.floor(np.min(z, axis=0) - margin_vox).astype(int)
    hi = np.ceil(np.max(z, axis=0) + margin_vox).astype(int) + 1
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.asarray(src.shape))
    sl = tuple(slice(lo[k], hi[k]) for k in range(3))
    roi = np.asarray(src[sl])
    cm = np.asarray(cur[sl])
    lm = np.asarray(leg[sl])
    union = cm | lm
    du = ndi.distance_transform_edt(~union, sampling=spacing)
    print("[expanded-field] ROI shape:", roi.shape, "margin_mm:", margin_mm)
    vessel = _frangi_3d(roi, spacing)

    kz = _xyz_to_zyx(ref, known) - lo[None, :]
    good = np.all((kz >= 0) & (kz < np.asarray(roi.shape)[None, :]), axis=1)
    kv = map_coordinates(vessel, kz[good].T, order=1, mode="nearest") if np.any(good) else np.array([0.03])
    thr = max(0.003, min(0.12, 0.30*float(np.percentile(kv, 20))))
    norm = max(float(np.median(kv))*1.5, 0.02)
    return {"lo": lo, "roi": roi, "cur": cm, "leg": lm, "du": du,
            "v": vessel, "thr": thr, "norm": norm,
            "margin_mm": float(margin_mm), "roi_shape": list(roi.shape)}


def _sample_field(ref, F, p):
    z = _xyz_to_zyx(ref, [p])[0] - F["lo"]
    shape = np.asarray(F["roi"].shape)
    if np.any(z < 1) or np.any(z > shape - 2):
        return None
    co = z[:, None]
    hu = float(map_coordinates(F["roi"], co, order=1, mode="nearest")[0])
    vv = float(map_coordinates(F["v"], co, order=1, mode="nearest")[0])
    dd = float(map_coordinates(F["du"], co, order=1, mode="nearest")[0])
    zi = np.rint(z).astype(int)
    c = bool(F["cur"][tuple(zi)])
    l = bool(F["leg"][tuple(zi)])
    return hu, vv, dd, c, l


def _beam(ref, F, aorta_tree, start, tangent,
          target_max_mm=SEARCH_MAX_MM, step=STEP_MM, beam_width=BEAM_WIDTH):
    origin = np.asarray(start, float)
    d0 = float(aorta_tree.query(origin)[0])
    states = [(0.0, [origin], _unit(tangent), d0)]
    reached = []
    best = states[0]
    rows = []
    nsteps = int(math.ceil(target_max_mm/step))

    for step_index in range(nsteps):
        counts = {
            "step_index": step_index,
            "arc_budget_mm": float((step_index+1)*step),
            "states_in": len(states),
            "proposals": 0,
            "reject_turn": 0,
            "reject_aorta_monotonic": 0,
            "reject_loop": 0,
            "reject_outside_field": 0,
            "reject_hu": 0,
            "reject_vesselness": 0,
            "reject_mask_gate": 0,
            "accepted_proposals": 0,
            "kept_states": 0,
            "reached_aorta": 0,
            "min_aorta_distance_mm": np.nan,
        }
        nxt = []
        for score, pts, t, prev_ad in states:
            for d in _cone_dirs(t):
                counts["proposals"] += 1
                if np.dot(d, t) < math.cos(math.radians(65)):
                    counts["reject_turn"] += 1
                    continue
                p = pts[-1] + step*d
                ad = float(aorta_tree.query(p)[0])
                if ad > prev_ad + 0.10:
                    counts["reject_aorta_monotonic"] += 1
                    continue
                if len(pts) > 5 and np.min(np.linalg.norm(np.asarray(pts[:-4])-p, axis=1)) < 0.60*step:
                    counts["reject_loop"] += 1
                    continue
                s = _sample_field(ref, F, p)
                if s is None:
                    counts["reject_outside_field"] += 1
                    continue
                hu, vv, du, c, l = s
                near = ad <= 1.3
                if not (80 <= hu <= 1400):
                    counts["reject_hu"] += 1
                    continue
                if vv < 0.42*F["thr"]:
                    counts["reject_vesselness"] += 1
                    continue
                if not (du <= 1.6 or near):
                    counts["reject_mask_gate"] += 1
                    continue
                vn = min(1.0, vv/F["norm"])
                support = 1.0 if c and l else (0.60 if c or l else 0.15 if du <= 1.6 else 0.0)
                improve = max(-0.25, prev_ad-ad)
                align = max(0.0, float(np.dot(d, t)))
                ns = score + 1.7*vn + 0.45*support + 0.45*align + 0.9*improve/max(step, 1e-6)
                nt = _unit(0.70*t + 0.30*d)
                st = (ns, pts+[p], nt, ad)
                nxt.append(st)
                counts["accepted_proposals"] += 1
                if (ad < best[3]-1e-9) or (abs(ad-best[3]) <= 1e-9 and ns > best[0]):
                    best = st
                if ad <= 0.75:
                    reached.append(st)
                    counts["reached_aorta"] += 1
        if not nxt:
            rows.append(counts)
            break
        nxt.sort(key=lambda x: x[0], reverse=True)
        keep, bins = [], set()
        for st in nxt:
            key = tuple(np.round(st[1][-1]/0.30).astype(int))
            if key in bins:
                continue
            bins.add(key)
            keep.append(st)
            if len(keep) >= beam_width:
                break
        states = keep
        counts["kept_states"] = len(states)
        counts["min_aorta_distance_mm"] = float(min(st[3] for st in states)) if states else np.nan
        rows.append(counts)
        if step_index % 10 == 0 or counts["reached_aorta"]:
            print(f"[expanded-field] step={step_index+1}/{nsteps} states={len(states)} "
                  f"min_aorta={counts['min_aorta_distance_mm']:.2f} reached={len(reached)}")
        if len(reached) >= 12:
            break

    chosen = None
    if reached:
        reached.sort(key=lambda x: (x[3], -x[0]))
        chosen = reached[0]
    return chosen, best, pd.DataFrame(rows), reached


def _plane(ref, src, c, t, half=5.5, step=0.20):
    t = _unit(t)
    u, v = _orth_basis(t)
    q = np.arange(-half, half+1e-9, step)
    yy, xx = np.meshgrid(q, q, indexing="ij")
    P = c + xx[...,None]*u + yy[...,None]*v
    return _sample(ref, src, P.reshape(-1,3)).reshape(len(q), len(q)), q


def _component_metrics(im, q):
    iy = ix = int(np.argmin(np.abs(q)))
    center = float(im[iy, ix])
    yy, xx = np.meshgrid(q, q, indexing="ij")
    rr = np.sqrt(xx*xx + yy*yy)
    thr = max(220.0, min(500.0, 0.55*center))
    bw = (im >= thr) & (rr <= 3.5)
    lab, _ = ndi.label(bw, np.ones((3,3), int))
    labels = []
    if lab[iy, ix] > 0:
        labels = [int(lab[iy, ix])]
    else:
        pts = np.argwhere(bw)
        if len(pts):
            dist = np.sqrt((q[pts[:,1]])**2 + (q[pts[:,0]])**2)
            j = int(np.argmin(dist))
            if dist[j] <= 1.0:
                labels = [int(lab[tuple(pts[j])])]
    if not labels:
        return {"center_hu": center, "threshold_hu": thr, "component_found": False,
                "radius_mm": np.nan, "centroid_offset_mm": np.inf,
                "axis_ratio": np.inf, "contrast_hu": -np.inf}
    mask = lab == labels[0]
    pts = np.argwhere(mask)
    xs = q[pts[:,1]]
    ys = q[pts[:,0]]
    cx = float(np.mean(xs)); cy = float(np.mean(ys))
    off = float(np.hypot(cx, cy))
    area = float(len(pts)*(0.20**2))
    rad = float(np.sqrt(area/np.pi))
    if len(pts) >= 4:
        C = np.cov(np.column_stack([xs, ys]).T)
        ev = np.linalg.eigvalsh(C)
        axis = float(np.sqrt(max(ev[-1],1e-6)/max(ev[0],1e-6)))
    else:
        axis = np.inf
    ring = (rr >= 3.5) & (rr <= 5.0)
    contrast = float(np.median(im[mask]) - np.median(im[ring])) if np.any(ring) else np.nan
    return {"center_hu": center, "threshold_hu": thr, "component_found": True,
            "radius_mm": rad, "centroid_offset_mm": off,
            "axis_ratio": axis, "contrast_hu": contrast}


def _fixed_plane_pass(m):
    return bool(m["component_found"] and m["center_hu"] >= 200
                and 0.55 <= m["radius_mm"] <= 3.2
                and m["centroid_offset_mm"] <= 1.10
                and m["axis_ratio"] <= 2.2
                and m["contrast_hu"] >= 40)


def _dense_qc(ref, src, path, label):
    p, q = _resample(path, PLANE_STEP_MM)
    if len(p) < 3:
        return p, q, pd.DataFrame()
    tt = np.gradient(p, axis=0)
    tt /= np.maximum(np.linalg.norm(tt, axis=1, keepdims=True), 1e-9)
    rows = []
    for i, (c, t, a) in enumerate(zip(p, tt, q)):
        im, grid = _plane(ref, src, c, t)
        m = _component_metrics(im, grid)
        m.update(index=i, arc_mm=float(a), label=label,
                 plane_pass=_fixed_plane_pass(m))
        rows.append(m)
    return p, q, pd.DataFrame(rows)


def _truncate_by_sustained_failure(df, nfail=SUSTAINED_FAIL_N):
    if df.empty:
        return {"first_sustained_failure_index": None, "accepted_last_index": -1,
                "accepted_arc_mm": 0.0, "accepted_plane_pass_fraction": 0.0,
                "accepted": False, "covers_full_path": False}
    passes = df["plane_pass"].astype(bool).to_numpy()
    cutoff = len(df)-1
    first_fail = None
    for i in range(0, len(passes)-nfail+1):
        if not np.any(passes[i:i+nfail]):
            first_fail = i
            cutoff = max(0, i-1)
            break
    accepted = df.iloc[:cutoff+1].copy()
    frac = float(accepted["plane_pass"].mean()) if len(accepted) else 0.0
    arc = float(accepted["arc_mm"].iloc[-1]) if len(accepted) else 0.0
    covers = bool(cutoff == len(df)-1)
    return {"first_sustained_failure_index": first_fail,
            "accepted_last_index": int(cutoff),
            "accepted_arc_mm": arc,
            "accepted_plane_pass_fraction": frac,
            "accepted": bool(arc >= MIN_EXTENSION_MM and frac >= MIN_PLANE_PASS_FRACTION),
            "covers_full_path": covers}


def _rca_plane_control(ref, src, rca, aorta_tree):
    rca_o, _, _ = _orient_endpoint_nearest_tree(rca, aorta_tree)
    rp, rq = _resample(rca_o, 0.8)
    keep = (rq >= 2.0) & (rq <= min(18.0, rq[-1]))
    rp = rp[keep]
    if len(rp) < 8:
        rp, _ = _resample(rca_o, 0.8)
    tt = np.gradient(rp, axis=0)
    tt /= np.maximum(np.linalg.norm(tt, axis=1, keepdims=True), 1e-9)
    rows = []
    for i, (c, t) in enumerate(zip(rp, tt)):
        im, g = _plane(ref, src, c, t)
        m = _component_metrics(im, g)
        m["plane_pass"] = _fixed_plane_pass(m)
        m["index"] = i
        rows.append(m)
    df = pd.DataFrame(rows)
    frac = float(df["plane_pass"].mean()) if len(df) else 0.0
    return df, bool(frac >= 0.75), frac, rca_o


def _path_source_metrics(ref, src, F, path):
    p, q = _resample(path, 0.20)
    z = _xyz_to_zyx(ref, p) - F["lo"][None,:]
    shape = np.asarray(F["roi"].shape)
    inside = np.all((z >= 0) & (z <= shape[None,:]-1), axis=1)
    hu = np.full(len(p), np.nan)
    vv = np.full(len(p), np.nan)
    du = np.full(len(p), np.nan)
    if np.any(inside):
        zz = z[inside].T
        hu[inside] = map_coordinates(F["roi"], zz, order=1, mode="nearest")
        vv[inside] = map_coordinates(F["v"], zz, order=1, mode="nearest")
        du[inside] = map_coordinates(F["du"], zz, order=1, mode="nearest")
    return pd.DataFrame({"arc_mm": q, "hu": hu, "vesselness": vv,
                         "distance_to_coronary_mask_mm": du})


def synthetic_expanded_field_self_test():
    start_distance = 29.88011340574657
    assert SEARCH_MAX_MM >= start_distance + PROSPECTIVE_MARGIN_MM
    d = pd.DataFrame({"arc_mm": np.arange(10)*0.4,
                      "plane_pass": [1,1,1,1,0,1,1,0,0,0]})
    t = _truncate_by_sustained_failure(d, 3)
    assert t["first_sustained_failure_index"] == 7
    assert t["accepted_last_index"] == 6
    return {"ok": True, "search_max_mm": SEARCH_MAX_MM,
            "field_margin_mm": FIELD_MARGIN_MM,
            "prospective_margin_mm": PROSPECTIVE_MARGIN_MM}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out/"run_state.json",
                {"status": "STARTED", "algorithm": ALGORITHM,
                 "baseline_commit": BASELINE})

    prior_summary_path = root / PRIOR_SUMMARY
    prior_path_file = root / PRIOR_ACCEPTED_PATH
    required = [
        prior_summary_path, prior_path_file,
        root/SOURCE_CACHE/"series7_int16.npy",
        root/SOURCE_CACHE/"series7_int16.json",
        root/MASTER, root/RCA_PATH, root/CUR, root/LEG, root/AORTA,
    ]
    for p in required:
        _req(p)

    prior = _load_json(prior_summary_path)
    master = _load_json(root/MASTER)
    if prior.get("status") != "PROXIMAL_TRUNK_CONTINUATION_QC_POSITIVE":
        raise RuntimeError(f"Unexpected prerequisite status: {prior.get('status')}")
    if prior.get("baseline_commit") != BASELINE:
        raise RuntimeError("Prerequisite baseline mismatch")
    prior_qc = prior.get("candidate_dense_qc", {})
    if not bool(prior_qc.get("accepted", False)):
        raise RuntimeError("Prerequisite continuation was not accepted by dense QC")

    ref, src, spacing = _source(root/SOURCE_CACHE)
    cur = _resample_mask(root/CUR, ref)
    leg = _resample_mask(root/LEG, ref)
    aorta = _resample_mask(root/AORTA, ref)
    aorta_tree, aorta_pts = _surface_tree(aorta, ref)
    rca = _load_path(root/RCA_PATH, ref)
    accepted = _load_path(prior_path_file, ref)

    start = np.asarray(accepted[-1], float)
    ap, aq = _resample(accepted, 0.20)
    look = max(0, len(ap)-1-int(round(2.0/0.20)))
    tangent = _unit(ap[-1] - ap[look])
    start_aorta = float(aorta_tree.query(start)[0])
    goal = np.asarray(aorta_tree.data[int(aorta_tree.query(start)[1])], float)
    budget = {
        "start_aorta_distance_mm": start_aorta,
        "max_search_mm": SEARCH_MAX_MM,
        "prospective_margin_mm": PROSPECTIVE_MARGIN_MM,
        "budget_minus_straight_line_mm": float(SEARCH_MAX_MM-start_aorta),
        "prospectively_adequate": bool(SEARCH_MAX_MM >= start_aorta + PROSPECTIVE_MARGIN_MM),
        "field_margin_mm": FIELD_MARGIN_MM,
        "previous_field_margin_mm": 7.0,
    }
    _write_json(out/"expanded_field_search_budget.json", budget)
    if not budget["prospectively_adequate"]:
        raise RuntimeError(f"Search budget inadequate: {budget}")

    print("Prior accepted continuation (mm):", prior_qc.get("accepted_arc_mm"))
    print("New endpoint distance to aorta (mm):", round(start_aorta, 3))
    print("Expanded field margin (mm):", FIELD_MARGIN_MM)

    rdf, rca_control_pass, rca_pass_fraction, rca_o = _rca_plane_control(
        ref, src, rca, aorta_tree)
    rdf.to_csv(out/"RCA_plane_qc_control.csv", index=False)
    print("RCA plane-QC control:", rca_control_pass, "pass_fraction=", round(rca_pass_fraction, 3))

    if not rca_control_pass:
        status = STATUS_RCA_FAIL
        summary = {
            "status": status, "algorithm": ALGORITHM, "baseline_commit": BASELINE,
            "master_status": master.get("status"), "master_modified": False,
            "prior_continuation_status": prior.get("status"),
            "RCA_plane_qc_control": {"plane_count": int(len(rdf)),
                                     "pass_fraction": rca_pass_fraction,
                                     "accepted": False},
            "search_budget": budget,
        }
        _write_json(out/"summary.json", summary)
        _write_json(out/"run_state.json",
                    {"status": "COMPLETE", "scientific_status": status,
                     "algorithm": ALGORITHM, "baseline_commit": BASELINE})
        return _finalize(out, summary)

    F = _expanded_field(ref, src, spacing, cur, leg, start, goal, accepted,
                        margin_mm=FIELD_MARGIN_MM)
    _write_json(out/"expanded_field_metadata.json",
                {"lo_zyx": F["lo"].tolist(), "roi_shape": F["roi_shape"],
                 "field_margin_mm": FIELD_MARGIN_MM,
                 "vesselness_threshold": float(F["thr"]),
                 "vesselness_normalizer": float(F["norm"])})

    chosen, best, attr, reached = _beam(ref, F, aorta_tree, start, tangent)
    attr.to_csv(out/"expanded_field_attrition.csv", index=False)

    state = chosen if chosen is not None else best
    path = np.asarray(state[1], float)
    pd.DataFrame(path, columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(
        out/"best_reacquisition_path.csv", index=False)
    profile = _path_source_metrics(ref, src, F, path)
    profile.to_csv(out/"best_reacquisition_source_profile.csv", index=False)

    pp, qq, qcdf = _dense_qc(ref, src, path, "expanded_field_reacquisition")
    qcdf.to_csv(out/"best_reacquisition_dense_plane_qc.csv", index=False)
    trunc = _truncate_by_sustained_failure(qcdf)
    last = int(trunc["accepted_last_index"])
    accepted_extension = pp[:last+1] if last >= 0 else np.empty((0,3))
    extension_positive = bool(trunc["accepted"])

    endpoint_aorta = float(aorta_tree.query(accepted_extension[-1])[0]) if len(accepted_extension) else np.nan
    reached_aorta = bool(chosen is not None)
    full_qc_to_aorta = bool(reached_aorta and trunc["covers_full_path"]
                            and trunc["accepted_plane_pass_fraction"] >= MIN_PLANE_PASS_FRACTION)

    rca_oriented, _, _ = _orient_endpoint_nearest_tree(rca, aorta_tree)
    rca_prox = np.asarray(rca_oriented[0], float)
    endpoint_rca_sep = (float(np.linalg.norm(path[-1]-rca_prox))
                        if reached_aorta else np.nan)
    rca_independent = bool(reached_aorta and endpoint_rca_sep >= RCA_ENDPOINT_SEPARATION_MM)

    if full_qc_to_aorta and rca_independent:
        status = STATUS_BRIDGE
    elif full_qc_to_aorta and not rca_independent:
        status = STATUS_RCA_ASSOC
    elif extension_positive:
        status = STATUS_EXTEND
    else:
        status = STATUS_FAIL

    if extension_positive:
        pd.DataFrame(accepted_extension,
                     columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(
            out/"accepted_endpoint_extension_candidate.csv", index=False)
        combined = np.vstack([accepted, accepted_extension[1:]])
        pd.DataFrame(combined,
                     columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(
            out/"combined_proximal_trunk_candidate.csv", index=False)

    plt.figure(figsize=(8.0, 4.5))
    if len(attr):
        plt.plot(attr["arc_budget_mm"], attr["min_aorta_distance_mm"], marker=".", ms=2)
    plt.axhline(0.75, lw=0.8)
    plt.xlabel("reacquisition search arc budget (mm)")
    plt.ylabel("minimum distance to aorta (mm)")
    plt.title("Expanded-field endpoint reacquisition")
    plt.tight_layout()
    plt.savefig(out/"01_expanded_field_aorta_distance.png", dpi=180)
    plt.close()

    if not qcdf.empty:
        plt.figure(figsize=(8.0, 4.5))
        plt.plot(qcdf["arc_mm"], qcdf["center_hu"], label="center HU")
        plt.axhline(200, ls="--", lw=0.8)
        plt.xlabel("reacquisition path arc (mm)")
        plt.ylabel("center HU")
        ax2 = plt.twinx()
        ax2.plot(qcdf["arc_mm"], qcdf["plane_pass"].astype(int), alpha=0.45)
        ax2.set_ylabel("plane pass")
        plt.title("Dense source-CCTA QC of endpoint reacquisition")
        plt.tight_layout()
        plt.savefig(out/"02_reacquisition_dense_qc_profile.png", dpi=180)
        plt.close()

        picks = np.linspace(0, len(pp)-1, min(12, len(pp))).astype(int)
        cols = 4
        rows = int(math.ceil(len(picks)/cols))
        fig, axes = plt.subplots(rows, cols, figsize=(14, 3.6*rows))
        axes = np.atleast_1d(axes).ravel()
        tt = np.gradient(pp, axis=0)
        tt /= np.maximum(np.linalg.norm(tt, axis=1, keepdims=True), 1e-9)
        for ax in axes[len(picks):]:
            ax.axis("off")
        for ax, ix in zip(axes, picks):
            im, g = _plane(ref, src, pp[ix], tt[ix])
            ax.imshow(im, cmap="gray", vmin=-100, vmax=900,
                      extent=[g[0],g[-1],g[-1],g[0]])
            ax.scatter([0],[0],s=14)
            row = qcdf.iloc[ix]
            ax.set_title(f"{row.arc_mm:.1f} mm | pass={bool(row.plane_pass)}")
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Expanded-field proximal-trunk endpoint reacquisition")
        plt.tight_layout()
        plt.savefig(out/"03_reacquisition_orthogonal_source_qc.png", dpi=180)
        plt.close()

    summary = {
        "status": status,
        "algorithm": ALGORITHM,
        "baseline_commit": BASELINE,
        "master_status": master.get("status"),
        "master_modified": False,
        "prior_continuation_status": prior.get("status"),
        "prior_accepted_continuation_mm": float(prior_qc.get("accepted_arc_mm", np.nan)),
        "RCA_plane_qc_control": {
            "plane_count": int(len(rdf)),
            "pass_fraction": rca_pass_fraction,
            "accepted": rca_control_pass,
        },
        "search_budget": budget,
        "expanded_field": {
            "margin_mm": FIELD_MARGIN_MM,
            "roi_shape": F["roi_shape"],
            "vesselness_threshold": float(F["thr"]),
            "reached_aorta": reached_aorta,
            "n_reached_states": int(len(reached)),
            "best_aorta_distance_mm": float(best[3]),
            "best_path_points": int(len(best[1])),
        },
        "dense_qc": {
            **trunc,
            "path_total_arc_mm": float(qq[-1]) if len(qq) else 0.0,
            "accepted_extension_endpoint_aorta_distance_mm": endpoint_aorta,
        },
        "posthoc_aortic_endpoint_distance_to_known_RCA_prox_mm": endpoint_rca_sep,
        "posthoc_RCA_independent": rca_independent,
        "candidate_role": (
            "source-supported continuation from the independently dense-QC proximal-trunk endpoint; "
            "not labeled LM and not added to frozen master"
        ),
        "scientific_change": (
            "starts at the independently QC-accepted 20-mm proximal-trunk endpoint and expands the "
            "source search-field margin from 7 to 18 mm; prior HU/vesselness/mask/turn/aorta-monotonic "
            "beam rules are retained"
        ),
        "scientific_boundary": (
            "A positive aortic bridge nominates a source-supported proximal-trunk-to-aorta path for "
            "visual adjudication. It does not establish clinical LM identity, LCX topology, or modify "
            "the frozen master."
        ),
    }
    _write_json(out/"summary.json", summary)
    _write_json(out/"run_state.json",
                {"status": "COMPLETE", "scientific_status": status,
                 "algorithm": ALGORITHM, "baseline_commit": BASELINE})
    return _finalize(out, summary)


def _finalize(out, summary):
    report = out/"OPENPLAQUE_PROXIMAL_TRUNK_EXPANDED_FIELD_REACQUISITION_REPORT.html"
    dq = summary.get("dense_qc", {})
    ef = summary.get("expanded_field", {})
    report.write_text(
        "<html><body><h1>OpenPlaque Proximal-Trunk Expanded-Field Reacquisition v1</h1>"
        f"<p><b>Status:</b> {summary.get('status')}</p>"
        f"<p>RCA plane-QC control accepted: {summary.get('RCA_plane_qc_control',{}).get('accepted')}.</p>"
        f"<p>Expanded field reached aorta: {ef.get('reached_aorta')}.</p>"
        f"<p>Best aorta distance: {ef.get('best_aorta_distance_mm')} mm.</p>"
        f"<p>Dense-QC accepted extension: {dq.get('accepted_arc_mm')} mm.</p>"
        f"<p>Frozen master unchanged; LM identity remains unresolved unless separately adjudicated.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    zpath = out/"OPENPLAQUE_PROXIMAL_TRUNK_EXPANDED_FIELD_REACQUISITION_RESULTS.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p != zpath and p.is_file():
                z.write(p, p.name)
    return {"summary": summary, "report": str(report), "zip": str(zpath)}
