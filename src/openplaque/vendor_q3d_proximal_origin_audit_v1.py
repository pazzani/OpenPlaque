from __future__ import annotations

import json
import math
import os
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from scipy import ndimage as ndi

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "vendor-q3d-proximal-origin-audit-v1.0"
OUTPUT_DIRNAME = "Vendor_Q3D_Proximal_Origin_Audit_v1"

SERIES = {
    "RCA": {"number": 1035, "folder": "32218", "description": "RCA Curved Range Radial Q3D(MT)"},
    "CX":  {"number": 1039, "folder": "32219", "description": "CX Curved Range Radial Q3D(MT)"},
    "LAD": {"number": 1043, "folder": "32220", "description": "LAD Curved Range Radial Q3D(MT)"},
}
SERIES7_FOLDER = "32210"

EXPECTED_FILES = 24
EXPECTED_ROWS = 512
EXPECTED_COLS = 512
MIN_RCA_SIDE_CONSISTENCY = 0.75
MIN_RCA_PROX_DISTAL_RATIO = 1.20
MIN_TARGET_SIDE_CONSISTENCY = 0.60
MIN_TARGET_PROX_DISTAL_RATIO = 1.10
MIN_TARGET_RCA_CALIBRATED_INDEX = 0.50
END_FRACTION = 0.18
CENTER_SEARCH_HALF = 110
WIDTH_HALF = 90
CENTER_BAND_HALF = 7

STATUS_BOTH = "Q3D_BOTH_LEFT_ORIGIN_SIGNATURES_PRESENT"
STATUS_LAD = "Q3D_LAD_ORIGIN_SIGNATURE_PRESENT"
STATUS_CX = "Q3D_CX_ORIGIN_SIGNATURE_PRESENT"
STATUS_NONE = "Q3D_NO_LEFT_ORIGIN_SIGNATURE"
STATUS_RCA_FAIL = "Q3D_RCA_ORIGIN_CONTROL_FAILED"


def _req(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")


def _natural_key(p):
    import re
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", Path(p).name)]


def _safe_float(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v, default=-1):
    try:
        return int(v)
    except Exception:
        return int(default)


def _safe_text(v, limit=500):
    if v is None:
        return ""
    if isinstance(v, bytes):
        return f"<bytes {len(v)}>"
    try:
        s = str(v)
    except Exception:
        return "<unprintable>"
    if len(s) > limit:
        return s[:limit] + "..."
    return s


def _dicom_array(ds):
    a = ds.pixel_array.astype(np.float32)
    slope = _safe_float(getattr(ds, "RescaleSlope", 1.0), 1.0)
    intercept = _safe_float(getattr(ds, "RescaleIntercept", 0.0), 0.0)
    return a * slope + intercept


def _series_files(folder):
    return sorted([p for p in Path(folder).iterdir() if p.is_file()], key=_natural_key)


def load_q3d_series(folder, expected_number, expected_description):
    files = _series_files(folder)
    if len(files) != EXPECTED_FILES:
        raise RuntimeError(f"Expected {EXPECTED_FILES} Q3D files in {folder}, found {len(files)}")
    datasets = []
    arrays = []
    for p in files:
        ds = pydicom.dcmread(str(p), force=True)
        if _safe_int(getattr(ds, "SeriesNumber", -1)) != int(expected_number):
            raise RuntimeError(f"Unexpected SeriesNumber in {p}: {getattr(ds, 'SeriesNumber', None)}")
        if str(expected_description).lower() not in str(getattr(ds, "SeriesDescription", "")).lower():
            raise RuntimeError(
                f"Unexpected Q3D description in {p}: {getattr(ds, 'SeriesDescription', '')}"
            )
        if _safe_int(getattr(ds, "Rows", -1)) != EXPECTED_ROWS or _safe_int(getattr(ds, "Columns", -1)) != EXPECTED_COLS:
            raise RuntimeError(f"Unexpected Q3D matrix in {p}")
        try:
            arr = _dicom_array(ds)
        except Exception as e:
            raise RuntimeError(
                "Q3D pixel decompression failed. Install pylibjpeg and pylibjpeg-libjpeg. "
                + str(e)
            ) from e
        datasets.append(ds)
        arrays.append(arr)
    return files, datasets, np.stack(arrays).astype(np.float32)


def _iter_elements(ds, prefix=""):
    for elem in ds:
        if elem.tag == (0x7FE0, 0x0010):
            continue
        path = f"{prefix}{elem.tag}"
        if elem.VR == "SQ":
            yield {
                "path": path,
                "tag": str(elem.tag),
                "keyword": elem.keyword or "",
                "name": elem.name,
                "vr": elem.VR,
                "private": bool(elem.tag.is_private),
                "value": f"<Sequence {len(elem.value)}>",
            }
            for i, item in enumerate(elem.value):
                yield from _iter_elements(item, prefix=f"{path}[{i}]/")
        else:
            val = elem.value
            if elem.VR in {"OB","OW","OF","OD","OL","OV","UN"}:
                try:
                    sval = f"<binary {len(val)}>"
                except Exception:
                    sval = "<binary>"
            else:
                sval = _safe_text(val)
            yield {
                "path": path,
                "tag": str(elem.tag),
                "keyword": elem.keyword or "",
                "name": elem.name,
                "vr": elem.VR,
                "private": bool(elem.tag.is_private),
                "value": sval,
            }


def tag_inventory(series_datasets):
    rows = []
    for vessel, dsets in series_datasets.items():
        bykey = defaultdict(list)
        for ds in dsets:
            for rec in _iter_elements(ds):
                key = (rec["path"], rec["tag"], rec["keyword"], rec["name"], rec["vr"], rec["private"])
                bykey[key].append(rec["value"])
        for key, vals in bykey.items():
            uniq = list(dict.fromkeys(vals))
            rows.append({
                "vessel": vessel,
                "path": key[0],
                "tag": key[1],
                "keyword": key[2],
                "name": key[3],
                "vr": key[4],
                "private": key[5],
                "n_images_with_tag": len(vals),
                "n_unique_values": len(uniq),
                "example_values": " || ".join(uniq[:4]),
            })
    return pd.DataFrame(rows)


def geometry_inventory(series_files, series_datasets):
    rows = []
    for vessel, dsets in series_datasets.items():
        for p, ds in zip(series_files[vessel], dsets):
            ps = getattr(ds, "PixelSpacing", None)
            iop = getattr(ds, "ImageOrientationPatient", None)
            ipp = getattr(ds, "ImagePositionPatient", None)
            rows.append({
                "vessel": vessel,
                "file": str(p),
                "instance_number": _safe_int(getattr(ds, "InstanceNumber", -1)),
                "sop_class_uid": _safe_text(getattr(ds, "SOPClassUID", "")),
                "sop_instance_uid": _safe_text(getattr(ds, "SOPInstanceUID", "")),
                "series_uid": _safe_text(getattr(ds, "SeriesInstanceUID", "")),
                "frame_of_reference_uid": _safe_text(getattr(ds, "FrameOfReferenceUID", "")),
                "image_type": _safe_text(getattr(ds, "ImageType", "")),
                "derivation_description": _safe_text(getattr(ds, "DerivationDescription", "")),
                "rows": _safe_int(getattr(ds, "Rows", -1)),
                "cols": _safe_int(getattr(ds, "Columns", -1)),
                "pixel_spacing": _safe_text(ps),
                "image_orientation_patient": _safe_text(iop),
                "image_position_patient": _safe_text(ipp),
                "slice_location": _safe_float(getattr(ds, "SliceLocation", np.nan)),
                "has_standard_plane_geometry": bool(
                    ps is not None and iop is not None and ipp is not None
                    and len(ps) == 2 and len(iop) == 6 and len(ipp) == 3
                ),
                "private_tag_count": int(sum(1 for e in ds if e.tag.is_private)),
            })
    return pd.DataFrame(rows)


def _extract_referenced_sops(ds):
    out = []
    def walk(item, path=""):
        for elem in item:
            if elem.tag == (0x7FE0, 0x0010):
                continue
            if elem.keyword == "ReferencedSOPInstanceUID":
                out.append((path + "/" + str(elem.tag), str(elem.value)))
            if elem.VR == "SQ":
                for i, child in enumerate(elem.value):
                    walk(child, path + "/" + str(elem.tag) + f"[{i}]")
    walk(ds)
    return out


def series7_sop_map(folder):
    rows = {}
    for p in _series_files(folder):
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            sop = str(getattr(ds, "SOPInstanceUID", ""))
            ipp = getattr(ds, "ImagePositionPatient", None)
            if sop and ipp is not None and len(ipp) == 3:
                rows[sop] = np.asarray([float(x) for x in ipp], float)
        except Exception:
            continue
    return rows


def reference_mapping(series_files, series_datasets, s7map):
    rows = []
    for vessel, dsets in series_datasets.items():
        for p, ds in zip(series_files[vessel], dsets):
            refs = _extract_referenced_sops(ds)
            matched = [s7map[uid] for _, uid in refs if uid in s7map]
            rec = {
                "vessel": vessel,
                "file": str(p),
                "instance_number": _safe_int(getattr(ds, "InstanceNumber", -1)),
                "referenced_sop_count": len(refs),
                "matched_series7_sop_count": len(matched),
            }
            if matched:
                m = np.stack(matched)
                rec.update({
                    "matched_lps_centroid_x_mm": float(m[:,0].mean()),
                    "matched_lps_centroid_y_mm": float(m[:,1].mean()),
                    "matched_lps_centroid_z_mm": float(m[:,2].mean()),
                    "matched_lps_z_min_mm": float(m[:,2].min()),
                    "matched_lps_z_max_mm": float(m[:,2].max()),
                })
            rows.append(rec)
    return pd.DataFrame(rows)


def _hu_like(arr):
    p1, p50, p99 = np.percentile(arr[np.isfinite(arr)], [1,50,99])
    return bool(p1 < -200 and p99 > 250 and p50 < 600)


def _bright_threshold(arr):
    finite = arr[np.isfinite(arr)]
    if not len(finite):
        return 0.0, False
    if _hu_like(arr):
        return 150.0, True
    # Q3D secondary images may be display-like rather than HU-like, and the
    # contrasted vessel can occupy only a few percent of pixels. Use the upper
    # tail rather than p85/p95 alone so sparse bright lumen is not swallowed by
    # the background distribution.
    p50, p90, p99 = np.percentile(finite, [50,90,99])
    if p99 <= p50 + 1e-6:
        return float(p99), False
    thr = max(p90, p50 + 0.45 * (p99 - p50))
    return float(thr), False


def _longest_run(b):
    b = np.asarray(b, bool)
    if not b.any():
        return 0
    z = np.r_[False, b, False].astype(np.int8)
    d = np.diff(z)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return int((ends - starts).max())


def _axis_score(arr, axis, threshold):
    n0, n1 = arr.shape
    if axis == "horizontal":
        c = n0 // 2
        band = arr[max(0,c-18):min(n0,c+19), :]
        prof = np.percentile(band, 82, axis=0)
    else:
        c = n1 // 2
        band = arr[:, max(0,c-18):min(n1,c+19)]
        prof = np.percentile(band, 82, axis=1)
    on = prof >= threshold
    return float(_longest_run(on)) / max(len(on), 1)


def detect_axis(arr):
    thr, hu = _bright_threshold(arr)
    hs = _axis_score(arr, "horizontal", thr)
    vs = _axis_score(arr, "vertical", thr)
    return ("horizontal" if hs >= vs else "vertical"), thr, hu, hs, vs


def _center_coordinate(arr, axis, threshold):
    nr, nc = arr.shape
    if axis == "horizontal":
        lo, hi = max(0,nr//2-CENTER_SEARCH_HALF), min(nr,nr//2+CENTER_SEARCH_HALF+1)
        central = arr[lo:hi, int(.25*nc):int(.75*nc)]
        persistence = (central >= threshold).mean(axis=1)
        coords = np.arange(lo,hi)
        prior = np.exp(-0.5*((coords-nr/2)/(CENTER_SEARCH_HALF*.65))**2)
        score = ndi.gaussian_filter1d(persistence, 2.0) * (0.65 + 0.35*prior)
        return int(coords[np.argmax(score)])
    lo, hi = max(0,nc//2-CENTER_SEARCH_HALF), min(nc,nc//2+CENTER_SEARCH_HALF+1)
    central = arr[int(.25*nr):int(.75*nr), lo:hi]
    persistence = (central >= threshold).mean(axis=0)
    coords = np.arange(lo,hi)
    prior = np.exp(-0.5*((coords-nc/2)/(CENTER_SEARCH_HALF*.65))**2)
    score = ndi.gaussian_filter1d(persistence, 2.0) * (0.65 + 0.35*prior)
    return int(coords[np.argmax(score)])


def _roi_component_metrics(mask, axis, center, side):
    nr, nc = mask.shape
    n = nc if axis == "horizontal" else nr
    e = max(16, int(round(END_FRACTION*n)))
    if axis == "horizontal":
        y0,y1=max(0,center-WIDTH_HALF),min(nr,center+WIDTH_HALF+1)
        sub = mask[y0:y1, :e] if side == "left" else mask[y0:y1, nc-e:nc]
        c0 = center - y0
        spine = np.zeros_like(sub, bool)
        spine[max(0,c0-CENTER_BAND_HALF):min(sub.shape[0],c0+CENTER_BAND_HALF+1),:] = True
        widths = sub.sum(axis=0)
    else:
        x0,x1=max(0,center-WIDTH_HALF),min(nc,center+WIDTH_HALF+1)
        sub = mask[:e, x0:x1] if side == "left" else mask[nr-e:nr, x0:x1]
        c0 = center - x0
        spine = np.zeros_like(sub, bool)
        spine[:,max(0,c0-CENTER_BAND_HALF):min(sub.shape[1],c0+CENTER_BAND_HALF+1)] = True
        widths = sub.sum(axis=1)

    lab,nlab=ndi.label(sub,structure=np.ones((3,3),np.uint8))
    best_area=0
    best_span=0
    for cid in range(1,nlab+1):
        cm=lab==cid
        if not np.any(cm & spine):
            continue
        area=int(cm.sum())
        if axis == "horizontal":
            span=int(np.max(cm.sum(axis=0))) if area else 0
        else:
            span=int(np.max(cm.sum(axis=1))) if area else 0
        if area>best_area:
            best_area,best_span=area,span
    return {
        "component_area": best_area,
        "component_area_fraction": float(best_area/max(sub.size,1)),
        "component_max_width_px": best_span,
        "end_width_p75_px": float(np.percentile(widths,75)) if len(widths) else 0.0,
        "end_bright_fraction": float(sub.mean()) if sub.size else 0.0,
    }


def view_metrics(arr, forced_axis=None):
    detected, thr, hu, hs, vs = detect_axis(arr)
    axis = forced_axis or detected
    center = _center_coordinate(arr, axis, thr)
    mask = arr >= thr

    if axis == "horizontal":
        widths = mask[max(0,center-WIDTH_HALF):min(mask.shape[0],center+WIDTH_HALF+1), :].sum(axis=0)
    else:
        widths = mask[:, max(0,center-WIDTH_HALF):min(mask.shape[1],center+WIDTH_HALF+1)].sum(axis=1)
    n=len(widths)
    mid=widths[int(.30*n):int(.70*n)]
    baseline=float(np.median(mid)) if len(mid) else 1.0
    baseline=max(baseline,1.0)

    left=_roi_component_metrics(mask,axis,center,"left")
    right=_roi_component_metrics(mask,axis,center,"right")
    for side,rec in (("left",left),("right",right)):
        rec["expansion_ratio"]=float(rec["end_width_p75_px"]/baseline)
        rec["origin_score"]=float(
            math.log1p(rec["expansion_ratio"])
            + 7.0*rec["component_area_fraction"]
            + 0.01*rec["component_max_width_px"]
        )

    return {
        "detected_axis":detected,
        "analysis_axis":axis,
        "threshold":float(thr),
        "hu_like":bool(hu),
        "horizontal_axis_score":float(hs),
        "vertical_axis_score":float(vs),
        "center_coordinate_px":int(center),
        "baseline_width_px":baseline,
        "left":left,
        "right":right,
    }


def analyze_views(arrays_by_vessel):
    prelim=[]
    for vessel,stack in arrays_by_vessel.items():
        for i,arr in enumerate(stack):
            m=view_metrics(arr)
            prelim.append((vessel,i,m["detected_axis"]))
    axis_counts=Counter(x[2] for x in prelim)
    renderer_axis=axis_counts.most_common(1)[0][0]

    rows=[]
    for vessel,stack in arrays_by_vessel.items():
        for i,arr in enumerate(stack):
            m=view_metrics(arr,forced_axis=renderer_axis)
            rows.append({
                "vessel":vessel,
                "view_index":i,
                "detected_axis":m["detected_axis"],
                "analysis_axis":m["analysis_axis"],
                "axis_agrees_with_renderer":m["detected_axis"]==renderer_axis,
                "threshold":m["threshold"],
                "hu_like":m["hu_like"],
                "horizontal_axis_score":m["horizontal_axis_score"],
                "vertical_axis_score":m["vertical_axis_score"],
                "center_coordinate_px":m["center_coordinate_px"],
                "baseline_width_px":m["baseline_width_px"],
                **{f"left_{k}":v for k,v in m["left"].items()},
                **{f"right_{k}":v for k,v in m["right"].items()},
            })
    return renderer_axis,pd.DataFrame(rows)


def summarize_origin_signatures(view_df, renderer_axis):
    rca=view_df[view_df.vessel=="RCA"].copy()
    left_med=float(rca.left_origin_score.median())
    right_med=float(rca.right_origin_score.median())
    proximal_side="left" if left_med>=right_med else "right"
    distal_side="right" if proximal_side=="left" else "left"

    rca_consistency=float((rca[f"{proximal_side}_origin_score"] > rca[f"{distal_side}_origin_score"]).mean())
    rca_prox=float(rca[f"{proximal_side}_origin_score"].median())
    rca_dist=float(rca[f"{distal_side}_origin_score"].median())
    rca_ratio=float(rca_prox/max(rca_dist,1e-6))
    rca_control=bool(rca_consistency>=MIN_RCA_SIDE_CONSISTENCY and rca_ratio>=MIN_RCA_PROX_DISTAL_RATIO)

    denom=max(rca_prox-rca_dist,1e-6)
    rows=[]
    target_positive={}
    for vessel in ("RCA","LAD","CX"):
        g=view_df[view_df.vessel==vessel]
        prox=g[f"{proximal_side}_origin_score"]
        dist=g[f"{distal_side}_origin_score"]
        consistency=float((prox>dist).mean())
        p=float(prox.median()); d=float(dist.median())
        ratio=float(p/max(d,1e-6))
        calibrated=float((p-rca_dist)/denom)
        positive = True if vessel=="RCA" else bool(
            rca_control
            and consistency>=MIN_TARGET_SIDE_CONSISTENCY
            and ratio>=MIN_TARGET_PROX_DISTAL_RATIO
            and calibrated>=MIN_TARGET_RCA_CALIBRATED_INDEX
        )
        if vessel!="RCA": target_positive[vessel]=positive
        rows.append({
            "vessel":vessel,
            "renderer_axis":renderer_axis,
            "rca_calibrated_proximal_side":proximal_side,
            "proximal_side_consistency":consistency,
            "median_proximal_origin_score":p,
            "median_distal_origin_score":d,
            "proximal_distal_ratio":ratio,
            "rca_calibrated_origin_index":calibrated,
            "origin_signature_positive":positive,
        })

    if not rca_control:
        status=STATUS_RCA_FAIL
    elif target_positive["LAD"] and target_positive["CX"]:
        status=STATUS_BOTH
    elif target_positive["LAD"]:
        status=STATUS_LAD
    elif target_positive["CX"]:
        status=STATUS_CX
    else:
        status=STATUS_NONE

    return pd.DataFrame(rows),{
        "status":status,
        "renderer_axis":renderer_axis,
        "rca_control_pass":rca_control,
        "rca_proximal_side":proximal_side,
        "rca_proximal_side_consistency":rca_consistency,
        "rca_proximal_distal_ratio":rca_ratio,
        "LAD_origin_signature_positive":bool(target_positive.get("LAD",False)),
        "CX_origin_signature_positive":bool(target_positive.get("CX",False)),
    }


def _profile_for_view(arr,axis,center,threshold):
    mask=arr>=threshold
    if axis=="horizontal":
        lo,hi=max(0,center-WIDTH_HALF),min(arr.shape[0],center+WIDTH_HALF+1)
        width=mask[lo:hi,:].sum(axis=0).astype(float)
        inten=np.percentile(arr[max(0,center-12):min(arr.shape[0],center+13),:],82,axis=0)
    else:
        lo,hi=max(0,center-WIDTH_HALF),min(arr.shape[1],center+WIDTH_HALF+1)
        width=mask[:,lo:hi].sum(axis=1).astype(float)
        inten=np.percentile(arr[:,max(0,center-12):min(arr.shape[1],center+13)],82,axis=1)
    return width,inten


def proximal_profile_similarity(arrays_by_vessel,view_df,proximal_side):
    profiles={}
    for vessel,stack in arrays_by_vessel.items():
        rows=view_df[view_df.vessel==vessel].sort_values("view_index")
        ws=[]; ins=[]
        for (_,r),arr in zip(rows.iterrows(),stack):
            w,i=_profile_for_view(arr,r.analysis_axis,int(r.center_coordinate_px),float(r.threshold))
            if proximal_side=="right":
                w=w[::-1]; i=i[::-1]
            w=w/max(np.percentile(w,95),1.0)
            denom=max(np.percentile(i,95)-np.percentile(i,5),1e-6)
            i=(i-np.percentile(i,5))/denom
            ws.append(w); ins.append(i)
        profiles[vessel]=(np.median(np.stack(ws),axis=0),np.median(np.stack(ins),axis=0))
    n=max(8,int(round(.30*len(profiles["LAD"][0]))))
    a=np.r_[profiles["LAD"][0][:n],profiles["LAD"][1][:n]]
    b=np.r_[profiles["CX"][0][:n],profiles["CX"][1][:n]]
    corr=float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else np.nan
    return profiles,corr


def _display_limits(arr):
    f=arr[np.isfinite(arr)]
    return tuple(float(x) for x in np.percentile(f,[2,98])) if len(f) else (0.0,1.0)


def plot_montage(vessel,stack,metrics,proximal_side,out):
    g=metrics[metrics.vessel==vessel].copy()
    score_col=f"{proximal_side}_origin_score"
    picks=list(g.nlargest(4,score_col).view_index.astype(int))
    med_idx=int((g[score_col]-g[score_col].median()).abs().idxmin())
    med_view=int(g.loc[med_idx].view_index)
    if med_view not in picks: picks.append(med_view)
    picks=picks[:5]
    fig,axes=plt.subplots(1,len(picks),figsize=(4*len(picks),4))
    if len(picks)==1: axes=[axes]
    for ax,idx in zip(axes,picks):
        arr=stack[idx]; row=g[g.view_index==idx].iloc[0]; vmin,vmax=_display_limits(arr)
        ax.imshow(arr,cmap="gray",vmin=vmin,vmax=vmax)
        c=int(row.center_coordinate_px)
        if row.analysis_axis=="horizontal":
            ax.axhline(c,linewidth=.8)
        else:
            ax.axvline(c,linewidth=.8)
        ax.set_title(f"{vessel} view {idx}\n{score_col}={row[score_col]:.2f}")
        ax.axis("off")
    fig.suptitle(f"{vessel} Q3D most informative radial views | RCA-calibrated proximal side={proximal_side}")
    fig.tight_layout(); fig.savefig(out,dpi=170); plt.close(fig)


def plot_profiles(profiles,proximal_side,out):
    fig,ax=plt.subplots(figsize=(10,5))
    for vessel,(w,_) in profiles.items():
        x=np.arange(len(w))
        ax.plot(x,w,label=vessel)
    ax.set_xlabel(f"Pixels from RCA-calibrated proximal side ({proximal_side})")
    ax.set_ylabel("Median normalized bright width")
    ax.set_title("Q3D median proximal width profiles")
    ax.legend()
    fig.tight_layout(); fig.savefig(out,dpi=170); plt.close(fig)


def run(dicom_root="/content/drive/MyDrive/CCTA/DICOM/3221",drive_root="/content/drive/MyDrive/OpenPlaque",output_dir=None):
    root=Path(dicom_root); out=Path(output_dir) if output_dir else Path(drive_root)/OUTPUT_DIRNAME
    out.mkdir(parents=True,exist_ok=True)
    _write_json(out/"run_state.json",{"status":"RUNNING","algorithm":ALGORITHM,"baseline":BASELINE})

    files_by={}; dsets_by={}; arrays_by={}
    for vessel,spec in SERIES.items():
        folder=_req(root/spec["folder"])
        files,dsets,arrays=load_q3d_series(folder,spec["number"],spec["description"])
        files_by[vessel]=files; dsets_by[vessel]=dsets; arrays_by[vessel]=arrays

    geom=geometry_inventory(files_by,dsets_by)
    geom.to_csv(out/"q3d_geometry_inventory.csv",index=False)

    tags=tag_inventory(dsets_by)
    tags.to_csv(out/"q3d_tag_inventory.csv",index=False)
    tags[tags.private].to_csv(out/"q3d_private_tag_inventory.csv",index=False)

    s7map=series7_sop_map(_req(root/SERIES7_FOLDER))
    refs=reference_mapping(files_by,dsets_by,s7map)
    refs.to_csv(out/"q3d_source_reference_mapping.csv",index=False)

    renderer_axis,views=analyze_views(arrays_by)
    views.to_csv(out/"q3d_view_metrics.csv",index=False)
    signatures,decision=summarize_origin_signatures(views,renderer_axis)
    signatures.to_csv(out/"q3d_origin_signatures.csv",index=False)

    profiles,corr=proximal_profile_similarity(arrays_by,views,decision["rca_proximal_side"])
    decision["LAD_CX_proximal_profile_correlation"]=corr

    for vessel in ("RCA","LAD","CX"):
        plot_montage(vessel,arrays_by[vessel],views,decision["rca_proximal_side"],out/f"QC_{vessel}_q3d_informative_views.png")
    plot_profiles(profiles,decision["rca_proximal_side"],out/"01_q3d_proximal_width_profiles.png")

    direct_geom={}
    for vessel in ("RCA","LAD","CX"):
        g=geom[geom.vessel==vessel]
        direct_geom[vessel]={
            "all_24_have_standard_plane_geometry":bool(g.has_standard_plane_geometry.all()),
            "frame_of_reference_uids":sorted([x for x in g.frame_of_reference_uid.unique().tolist() if x]),
            "matched_series7_reference_count_total":int(refs[refs.vessel==vessel].matched_series7_sop_count.sum()),
        }

    summary={
        "status":decision["status"],
        "algorithm":ALGORITHM,
        "baseline_commit":BASELINE,
        "series":SERIES,
        "n_views_per_vessel":EXPECTED_FILES,
        "renderer_axis":renderer_axis,
        "decision":decision,
        "origin_signatures":signatures.to_dict("records"),
        "direct_geometry":direct_geom,
        "scientific_boundary":(
            "Vendor Q3D is same-exam derived evidence, not an independent acquisition. "
            "The 24 files are treated as radial curved views, not a physical 3-D stack. "
            "Pixel-space origin signatures are calibrated to the RCA positive control. "
            "No clinical LM/LCX/OM identity is established and the frozen master is not modified."
        ),
        "clinical_LM_identity_established":False,
        "clinical_LCX_OM_identity_established":False,
        "master_anatomy_modified":False,
    }
    _write_json(out/"decision.json",decision)
    _write_json(out/"summary.json",summary)

    report=out/"OPENPLAQUE_VENDOR_Q3D_PROXIMAL_ORIGIN_AUDIT_V1_REPORT.html"
    report.write_text(
        "<html><body><h1>OpenPlaque Vendor Q3D Proximal Origin Audit v1</h1>"
        f"<p><b>Status:</b> {decision['status']}</p>"
        "<p>RCA 1035 is the positive control; LAD 1043 and CX 1039 are target curved/radial Q3D series.</p>"
        "<p>The 24 files per vessel are not treated as a 3-D stack.</p>"
        "<h2>Origin signatures</h2>"+signatures.to_html(index=False)+
        "<h2>DICOM geometry summary</h2>"+pd.DataFrame([
            {"vessel":k,**v} for k,v in direct_geom.items()
        ]).to_html(index=False)+
        "<h2>Proximal width profiles</h2><img src='01_q3d_proximal_width_profiles.png' width='950'>"+
        "".join(f"<h2>{v}</h2><img src='QC_{v}_q3d_informative_views.png' width='1200'>" for v in ("RCA","LAD","CX"))+
        "</body></html>",encoding="utf-8"
    )

    _write_json(out/"run_state.json",{"status":"COMPLETE","algorithm":ALGORITHM,"result_status":decision["status"]})
    archive=out/"OPENPLAQUE_VENDOR_Q3D_PROXIMAL_ORIGIN_AUDIT_V1_RESULTS.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p!=archive:
                z.write(p,p.name)
    return summary


def synthetic_self_test():
    arr=np.full((512,512),-100.0,np.float32)
    arr[250:263,70:470]=350.0
    arr[205:305,0:110]=350.0
    m=view_metrics(arr,forced_axis="horizontal")
    assert m["left"]["origin_score"] > m["right"]["origin_score"]
    assert m["left"]["expansion_ratio"] > m["right"]["expansion_ratio"]
    return {"ok":True,"left_score":m["left"]["origin_score"],"right_score":m["right"]["origin_score"]}
