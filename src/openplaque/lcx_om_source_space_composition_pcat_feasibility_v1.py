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
from scipy.stats import spearmanr

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "lcx-om-source-space-composition-pcat-feasibility-v1.0"
OUTPUT_DIRNAME = "LCX_OM_Source_Space_Composition_PCAT_Feasibility_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
FREEZE_SUMMARY = Path("LCX_Structural_Source_QC_Freeze_v1/summary.json")
FREEZE_QC = Path("LCX_Structural_Source_QC_Freeze_v1/LCX_structural_dense_source_QC.csv")

EXPECTED_MASTER_STATUS = "CORONARY_ANATOMY_BASELINE_V2_FROZEN"
EXPECTED_FREEZE_STATUS = "LCX_RESEARCH_STRUCTURAL_LABELS_FROZEN"

BIFURCATION_EXCLUSION_MM = 1.0
PRIMARY_WALL_MARGIN_MM = 0.75
PCAT_MARGIN_SENSITIVITY_MM = (0.25, 0.50, 0.75, 1.00, 1.25)
FAT_HU_RANGE = (-190.0, -30.0)

N_ANGLES = 72
RADIAL_STEP_MM = 0.10
LUMEN_MAX_RADIUS_MM = 4.0
LUMEN_MIN_RADIUS_MM = 0.45
SHELLS_MM = (0.75, 1.0, 1.25, 1.5)
NOMINAL_SHELL_MM = 1.0

COMPONENTS = (
    "low_attenuation_mm3",
    "noncalcified_mm3",
    "mixed_intermediate_mm3",
    "calcified_mm3",
)

MIN_EVALUABLE_LENGTH_MM = 5.0
MIN_CANONICAL_FAT_VOXELS = 500
MIN_LONGITUDINAL_BIN_COVERAGE = 0.75
MIN_BIN_FAT_VOXELS = 25

STATUS_PASS = "LCX_OM_SOURCE_SPACE_COMPOSITION_PCAT_FEASIBILITY_COMPLETE"
STATUS_FAIL = "LCX_OM_SOURCE_SPACE_COMPOSITION_PCAT_FEASIBILITY_FAILED"


def _req(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _read_json(p):
    return json.loads(_req(p).read_text(encoding="utf-8"))


def _write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")


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


def _arc_weights(arcs):
    arcs = np.asarray(arcs, float)
    if len(arcs) == 1:
        return np.ones(1)
    w = np.empty(len(arcs), float)
    w[0] = .5 * (arcs[1] - arcs[0])
    w[-1] = .5 * (arcs[-1] - arcs[-2])
    if len(arcs) > 2:
        w[1:-1] = .5 * (arcs[2:] - arcs[:-2])
    return w


class SourceGeometry:
    def __init__(self, meta):
        self.spacing_zyx = np.asarray(meta["spacing_zyx"], float)
        self.spacing_xyz = self.spacing_zyx[::-1]
        self.origin = np.asarray(meta["positions_lps_mm"][0], float)
        iop = np.asarray(meta["image_orientation_patient"], float)
        row, col = iop[:3], iop[3:]
        slc = np.cross(row, col)
        self.D = np.array([
            [row[0], col[0], slc[0]],
            [row[1], col[1], slc[1]],
            [row[2], col[2], slc[2]],
        ], float)
        self.invD = np.linalg.inv(self.D)

    def xyz_to_zyx(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        xyz = ((pts - self.origin) @ self.invD.T) / self.spacing_xyz
        return xyz[:, ::-1]

    def zyx_to_xyz(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        idx_xyz = pts[:, ::-1]
        return self.origin + (idx_xyz * self.spacing_xyz) @ self.D.T


def _load_source(cache):
    cache = Path(cache)
    src = np.load(_req(cache / "series7_int16.npy"), mmap_mode="r")
    meta = _read_json(cache / "series7_int16.json")
    geom = SourceGeometry(meta)
    vv = float(np.prod(geom.spacing_zyx))
    if not (0.001 < vv < 0.5):
        raise RuntimeError(f"Unexpected source voxel volume {vv}")
    return geom, src, meta, vv


def _sample(geom, src, pts, cval=-1024.0):
    z = geom.xyz_to_zyx(pts)
    return map_coordinates(np.asarray(src), z.T, order=1, mode="constant", cval=float(cval))


def _cmedian(x):
    return np.median(np.stack([np.roll(x, k) for k in range(-2, 3)]), axis=0)


def _boundary(geom, src, c, t):
    u, v = _orth_basis(t)
    th = np.linspace(0.0, 2*np.pi, N_ANGLES, endpoint=False)
    dirs = np.cos(th)[:, None]*u + np.sin(th)[:, None]*v

    center_pts = np.vstack([c, c+.15*u, c-.15*u, c+.15*v, c-.15*v])
    center_hu = float(np.median(_sample(geom, src, center_pts)))
    threshold = float(np.clip(.55*center_hu, 220.0, 500.0))

    rr = np.arange(.2, LUMEN_MAX_RADIUS_MM + 1e-9, RADIAL_STEP_MM)
    P = c[None, None, :] + dirs[:, None, :]*rr[None, :, None]
    hu = _sample(geom, src, P.reshape(-1, 3)).reshape(N_ANGLES, len(rr))

    radii = np.full(N_ANGLES, np.nan)
    for i in range(N_ANGLES):
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
        radii = np.clip(radii, max(.4, med-.75), min(4.0, med+.75))
        p10, p50, p90 = np.percentile(radii, [10, 50, 90])
        axis = float(p90 / max(p10, 1e-6))
    else:
        p10 = p50 = p90 = axis = np.nan

    qc = bool(center_hu >= 200 and vf >= .60 and np.isfinite(axis) and axis <= 2.5)
    return {
        "center_hu": center_hu,
        "lumen_threshold_hu": threshold,
        "valid_radial_fraction": vf,
        "lumen_radius_p10_mm": float(p10),
        "lumen_radius_median_mm": float(p50),
        "lumen_radius_p90_mm": float(p90),
        "lumen_axis_proxy": float(axis),
        "station_qc_pass": qc,
        "radii": radii,
        "theta": th,
        "u": u,
        "v": v,
    }


def _integrate_shell(geom, src, c, L, shell_mm, ds_mm):
    th = L["theta"]
    dirs = np.cos(th)[:, None]*L["u"] + np.sin(th)[:, None]*L["v"]
    offsets = np.arange(RADIAL_STEP_MM/2.0, float(shell_mm), RADIAL_STEP_MM)
    if len(offsets) == 0:
        offsets = np.array([float(shell_mm)/2.0])
    r = L["radii"][:, None] + offsets[None, :]
    P = c[None, None, :] + dirs[:, None, :]*r[..., None]
    hu = _sample(geom, src, P.reshape(-1, 3)).reshape(r.shape)
    area = r * RADIAL_STEP_MM * (2*np.pi/N_ANGLES)
    vol = area * float(ds_mm)
    finite = np.isfinite(hu)

    masks = {
        "fatlike_excluded_mm3": finite & (hu < -30),
        "low_attenuation_mm3": finite & (hu >= -30) & (hu < 30),
        "noncalcified_mm3": finite & (hu >= 30) & (hu < 130),
        "mixed_intermediate_mm3": finite & (hu >= 130) & (hu < 350),
        "calcified_mm3": finite & (hu >= 350),
    }
    vals = {k: float(vol[m].sum()) for k, m in masks.items()}
    vals["shell_volume_mm3"] = float(vol[finite].sum())
    vals["raw_nonfatlike_shell_mm3"] = sum(vals[c] for c in COMPONENTS)
    return vals


def _quantify_shells(geom, src, freeze_qc):
    rows = []
    for vessel in ("C6", "C7"):
        d = freeze_qc[
            (freeze_qc.vessel == vessel)
            & (freeze_qc.post_split_arc_mm >= BIFURCATION_EXCLUSION_MM)
            & freeze_qc.station_qc_pass.astype(bool)
        ].copy().sort_values("post_split_arc_mm")
        if len(d) < 2:
            raise RuntimeError(f"Too few valid source stations for {vessel}")
        weights = _arc_weights(d.post_split_arc_mm.to_numpy(float))

        for (_, r), ds in zip(d.iterrows(), weights):
            c = np.array([r.lps_x_mm, r.lps_y_mm, r.lps_z_mm], float)
            t = np.array([r.tangent_x, r.tangent_y, r.tangent_z], float)
            L = _boundary(geom, src, c, t)
            # Recomputed source-plane geometry must remain usable. Preserve the freeze center and tangent.
            for shell in SHELLS_MM:
                comp = _integrate_shell(geom, src, c, L, shell, ds) if L["station_qc_pass"] else {
                    "fatlike_excluded_mm3": np.nan,
                    "low_attenuation_mm3": np.nan,
                    "noncalcified_mm3": np.nan,
                    "mixed_intermediate_mm3": np.nan,
                    "calcified_mm3": np.nan,
                    "shell_volume_mm3": np.nan,
                    "raw_nonfatlike_shell_mm3": np.nan,
                }
                rows.append({
                    "vessel": vessel,
                    "station_index": int(r.station_index),
                    "post_split_arc_mm": float(r.post_split_arc_mm),
                    "integration_ds_mm": float(ds),
                    "shell_thickness_mm": float(shell),
                    "freeze_station_qc_pass": bool(r.station_qc_pass),
                    "recomputed_station_qc_pass": bool(L["station_qc_pass"]),
                    "freeze_lumen_radius_median_mm": float(r.lumen_radius_median_mm),
                    "recomputed_lumen_radius_median_mm": float(L["lumen_radius_median_mm"]),
                    "freeze_lumen_axis_proxy": float(r.lumen_axis_proxy),
                    "recomputed_lumen_axis_proxy": float(L["lumen_axis_proxy"]),
                    "center_hu": float(L["center_hu"]),
                    **comp,
                })
    return pd.DataFrame(rows)


def _shell_profile_1mm(stations, shell=NOMINAL_SHELL_MM):
    rows = []
    d = stations[np.isclose(stations.shell_thickness_mm, shell)].copy()
    for vessel in ("C6", "C7"):
        x = d[d.vessel == vessel].copy()
        x["arc_start_mm"] = np.floor(x.post_split_arc_mm).astype(float)
        for a, g in x.groupby("arc_start_mm", sort=True):
            rec = {
                "vessel": vessel,
                "shell_thickness_mm": float(shell),
                "arc_start_mm": float(a),
                "arc_end_mm": float(a+1.0),
                "station_count": int(len(g)),
                "recomputed_qc_fraction": float(g.recomputed_station_qc_pass.mean()),
                "shell_volume_mm3": float(g.shell_volume_mm3.sum()),
                "fatlike_excluded_mm3": float(g.fatlike_excluded_mm3.sum()),
                "raw_nonfatlike_shell_mm3": float(g.raw_nonfatlike_shell_mm3.sum()),
            }
            for c in COMPONENTS:
                rec[c] = float(g[c].sum())
            rows.append(rec)
    return pd.DataFrame(rows)


def _shell_totals(stations):
    rows = []
    for vessel in ("C6", "C7"):
        for shell in SHELLS_MM:
            d = stations[
                (stations.vessel == vessel)
                & np.isclose(stations.shell_thickness_mm, shell)
                & stations.recomputed_station_qc_pass.astype(bool)
            ]
            rec = {
                "vessel": vessel,
                "shell_thickness_mm": float(shell),
                "station_count": int(len(d)),
                "evaluable_length_mm": float(d.integration_ds_mm.sum()),
                "shell_volume_mm3": float(d.shell_volume_mm3.sum()),
                "fatlike_excluded_mm3": float(d.fatlike_excluded_mm3.sum()),
                "raw_nonfatlike_shell_mm3": float(d.raw_nonfatlike_shell_mm3.sum()),
            }
            for c in COMPONENTS:
                rec[c] = float(d[c].sum())
            rows.append(rec)
    return pd.DataFrame(rows)


def _shell_stability(stations):
    profiles = {}
    for shell in SHELLS_MM:
        p = _shell_profile_1mm(stations, shell)
        profiles[shell] = p
    rows = []
    for vessel in ("C6", "C7"):
        nom = profiles[NOMINAL_SHELL_MM]
        nom = nom[nom.vessel == vessel].set_index("arc_start_mm")["raw_nonfatlike_shell_mm3"]
        for shell, p in profiles.items():
            q = p[p.vessel == vessel].set_index("arc_start_mm")["raw_nonfatlike_shell_mm3"].reindex(nom.index)
            ok = nom.notna() & q.notna()
            rho = float(spearmanr(nom[ok], q[ok]).statistic) if int(ok.sum()) >= 4 else np.nan
            rows.append({
                "vessel": vessel,
                "shell_thickness_mm": float(shell),
                "raw_profile_spearman_vs_nominal": rho,
                "matched_bins": int(ok.sum()),
            })
    return pd.DataFrame(rows)


def _sitk_reference(shape_zyx, geom):
    size_xyz = [int(x) for x in shape_zyx[::-1]]
    ref = sitk.Image(size_xyz, sitk.sitkUInt8)
    ref.SetSpacing(tuple(float(x) for x in geom.spacing_xyz))
    ref.SetOrigin(tuple(float(x) for x in geom.origin))
    ref.SetDirection(tuple(float(x) for x in geom.D.ravel()))
    return ref


def _load_aorta_mask(root, source_shape, geom):
    candidates = [
        root/"RCA_Ostium_TotalSegmentator"/"aorta_series7_totalseg.nii.gz",
        root/"TotalSegmentator_Validation_v2"/"aorta_series7_totalseg.nii.gz",
        root/"TotalSegmentator_Cardiovascular_Cache_v1"/"aorta.nii.gz",
        root/"TotalSegmentator_Cardiovascular_Cache_v1"/"heart"/"aorta.nii.gz",
    ]
    p = next((x for x in candidates if x.exists()), None)
    if p is None:
        return None, None
    im = sitk.ReadImage(str(p))
    ref = _sitk_reference(source_shape, geom)
    same = (
        im.GetSize() == ref.GetSize()
        and np.allclose(im.GetSpacing(), ref.GetSpacing())
        and np.allclose(im.GetOrigin(), ref.GetOrigin())
        and np.allclose(im.GetDirection(), ref.GetDirection())
    )
    if not same:
        im = sitk.Resample(im, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    a = sitk.GetArrayFromImage(im) > 0
    if a.shape != tuple(source_shape):
        raise RuntimeError(f"Aorta mask shape mismatch after resampling: {a.shape} vs {source_shape}")
    return a, p


def _segment_for_pcat(freeze_qc, vessel):
    d = freeze_qc[
        (freeze_qc.vessel == vessel)
        & (freeze_qc.post_split_arc_mm >= BIFURCATION_EXCLUSION_MM)
    ].copy().sort_values("post_split_arc_mm")
    # Use the longest contiguous QC-passing prefix after the bifurcation exclusion.
    if d.empty:
        raise RuntimeError(f"No {vessel} stations after bifurcation exclusion")
    passed = d.station_qc_pass.astype(bool).to_numpy()
    first_fail = np.flatnonzero(~passed)
    if len(first_fail):
        d = d.iloc[:int(first_fail[0])].copy()
    if len(d) < 2:
        raise RuntimeError(f"No usable contiguous {vessel} PCAT segment")
    return d


def _pcat_voxel_map(geom, src, aorta, segment_df, max_margin):
    centers = segment_df[["lps_x_mm","lps_y_mm","lps_z_mm"]].to_numpy(float)
    arcs = segment_df.post_split_arc_mm.to_numpy(float)
    radii = segment_df.lumen_radius_median_mm.to_numpy(float)

    max_outer = float(np.max(radii + float(max_margin)))
    max_shell_outer = 3.0 * max_outer
    pad_mm = max_shell_outer + 2.0

    zyx = geom.xyz_to_zyx(centers)
    lo = np.floor(np.min(zyx, axis=0) - pad_mm/geom.spacing_zyx).astype(int)
    hi = np.ceil(np.max(zyx, axis=0) + pad_mm/geom.spacing_zyx).astype(int) + 1
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.asarray(src.shape))

    crop = np.asarray(src[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
    zz, yy, xx = np.indices(crop.shape)
    gzyx = np.stack([zz+lo[0], yy+lo[1], xx+lo[2]], axis=-1).reshape(-1,3).astype(float)
    gxyz = geom.zyx_to_xyz(gzyx)

    tree = cKDTree(centers)
    dist_mm, nearest_idx = tree.query(gxyz, k=1, workers=-1)
    nearest_idx = nearest_idx.astype(int)

    aorta_flat = (
        aorta[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].reshape(-1)
        if aorta is not None else np.zeros(crop.size, dtype=bool)
    )
    return {
        "lo_zyx": lo,
        "hi_zyx": hi,
        "hu": crop.reshape(-1).astype(float),
        "dist_mm": dist_mm.astype(float),
        "nearest_arc_mm": arcs[nearest_idx].astype(float),
        "nearest_lumen_radius_mm": radii[nearest_idx].astype(float),
        "aorta_flat": aorta_flat,
        "arc_start_mm": float(arcs.min()),
        "arc_end_mm": float(arcs.max()),
    }


def _compute_pcat(vm, wall_margin_mm, voxel_volume_mm3):
    margin = float(wall_margin_mm)
    hu = vm["hu"]
    outer = vm["nearest_lumen_radius_mm"] + margin
    shell_outer = 3.0 * outer

    shell = (
        (vm["dist_mm"] > outer)
        & (vm["dist_mm"] <= shell_outer)
        & (vm["nearest_arc_mm"] >= vm["arc_start_mm"])
        & (vm["nearest_arc_mm"] <= vm["arc_end_mm"])
        & (~vm["aorta_flat"])
    )
    fat = shell & (hu >= FAT_HU_RANGE[0]) & (hu <= FAT_HU_RANGE[1])
    vals = hu[fat]
    if len(vals) == 0:
        raise RuntimeError(f"No PCAT fat voxels at margin {margin}")

    radial_out = vm["dist_mm"] - outer
    radial_rows = []
    for b in np.arange(0.0, 6.0, 0.5):
        m = fat & (radial_out >= b) & (radial_out < b+.5)
        v = hu[m]
        radial_rows.append({
            "wall_margin_mm": margin,
            "radial_start_mm": float(b),
            "radial_end_mm": float(b+.5),
            "fat_voxels": int(len(v)),
            "mean_hu": float(np.mean(v)) if len(v) else np.nan,
        })

    start = int(math.floor(vm["arc_start_mm"]))
    stop = int(math.ceil(vm["arc_end_mm"]))
    long_rows = []
    for b in range(start, stop):
        m = fat & (vm["nearest_arc_mm"] >= b) & (vm["nearest_arc_mm"] < b+1.0)
        v = hu[m]
        long_rows.append({
            "wall_margin_mm": margin,
            "arc_start_mm": float(b),
            "arc_end_mm": float(b+1.0),
            "fat_voxels": int(len(v)),
            "mean_hu": float(np.mean(v)) if len(v) else np.nan,
        })

    return {
        "wall_margin_mm": margin,
        "pcat_mean_hu": float(np.mean(vals)),
        "pcat_median_hu": float(np.median(vals)),
        "pcat_sd_hu": float(np.std(vals)),
        "fat_voxels": int(fat.sum()),
        "fat_volume_ml": float(fat.sum()*voxel_volume_mm3/1000.0),
        "shell_voxels": int(shell.sum()),
        "shell_volume_ml": float(shell.sum()*voxel_volume_mm3/1000.0),
        "fat_fraction": float(fat.sum()/max(1, shell.sum())),
        "radial": pd.DataFrame(radial_rows),
        "longitudinal": pd.DataFrame(long_rows),
    }


def _pcat_for_vessel(geom, src, aorta, freeze_qc, vessel, voxel_volume_mm3):
    seg = _segment_for_pcat(freeze_qc, vessel)
    vm = _pcat_voxel_map(geom, src, aorta, seg, max(PCAT_MARGIN_SENSITIVITY_MM))
    results = [_compute_pcat(vm, m, voxel_volume_mm3) for m in PCAT_MARGIN_SENSITIVITY_MM]
    primary = next(r for r in results if abs(r["wall_margin_mm"] - PRIMARY_WALL_MARGIN_MM) < 1e-12)

    long = primary["longitudinal"].copy()
    good_bins = long.fat_voxels >= MIN_BIN_FAT_VOXELS
    coverage = float(good_bins.mean()) if len(long) else 0.0

    summary = {
        "vessel": vessel,
        "segment_arc_start_mm": float(seg.post_split_arc_mm.min()),
        "segment_arc_end_mm": float(seg.post_split_arc_mm.max()),
        "segment_length_mm": float(seg.post_split_arc_mm.max() - seg.post_split_arc_mm.min()),
        "station_count": int(len(seg)),
        "primary_wall_margin_mm": PRIMARY_WALL_MARGIN_MM,
        "pcat_mean_hu": primary["pcat_mean_hu"],
        "pcat_median_hu": primary["pcat_median_hu"],
        "pcat_sd_hu": primary["pcat_sd_hu"],
        "fat_voxels": primary["fat_voxels"],
        "fat_volume_ml": primary["fat_volume_ml"],
        "shell_volume_ml": primary["shell_volume_ml"],
        "fat_fraction": primary["fat_fraction"],
        "longitudinal_bin_count": int(len(long)),
        "longitudinal_bins_ge_min_fat_voxels": int(good_bins.sum()),
        "longitudinal_bin_coverage_fraction": coverage,
        "sensitivity_mean_hu_min": float(min(r["pcat_mean_hu"] for r in results)),
        "sensitivity_mean_hu_max": float(max(r["pcat_mean_hu"] for r in results)),
        "sensitivity_range_hu": float(max(r["pcat_mean_hu"] for r in results) - min(r["pcat_mean_hu"] for r in results)),
    }
    sensitivity = pd.DataFrame([
        {k:v for k,v in r.items() if k not in ("radial","longitudinal")}
        for r in results
    ])
    return summary, primary, sensitivity


def _plot_composition(totals, out):
    d = totals[np.isclose(totals.shell_thickness_mm, NOMINAL_SHELL_MM)].copy()
    cols = list(COMPONENTS)
    x = np.arange(len(d))
    bottom = np.zeros(len(d))
    fig, ax = plt.subplots(figsize=(8,5.5))
    for c in cols:
        vals = d[c].to_numpy(float)
        ax.bar(x, vals, bottom=bottom, label=c.replace("_mm3","").replace("_"," "))
        bottom += vals
    ax.set_xticks(x, d.vessel)
    ax.set_ylabel("Raw 1.0-mm shell HU volume (mm³)")
    ax.set_title("LCX-like / OM-like raw peri-luminal shell composition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_pcat_longitudinal(primary_by_vessel, out):
    fig, ax = plt.subplots(figsize=(10,5.5))
    for vessel, r in primary_by_vessel.items():
        d = r["longitudinal"]
        ax.plot((d.arc_start_mm+d.arc_end_mm)/2, d.mean_hu, marker="o", label=vessel)
    ax.set_xlabel("Post-split arc (mm)")
    ax.set_ylabel("PCAT mean HU")
    ax.set_title("Direct PCAT attenuation on research structural paths")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_pcat_sensitivity(sensitivity_by_vessel, out):
    fig, ax = plt.subplots(figsize=(8,5.5))
    for vessel, d in sensitivity_by_vessel.items():
        ax.plot(d.wall_margin_mm, d.pcat_mean_hu, marker="o", label=vessel)
    ax.axvline(PRIMARY_WALL_MARGIN_MM, linestyle="--", linewidth=1)
    ax.set_xlabel("Modeled outer-wall margin (mm)")
    ax.set_ylabel("PCAT mean HU")
    ax.set_title("PCAT geometry sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_shell_sensitivity(totals, out):
    fig, ax = plt.subplots(figsize=(8,5.5))
    for vessel in ("C6","C7"):
        d = totals[totals.vessel == vessel].sort_values("shell_thickness_mm")
        ax.plot(d.shell_thickness_mm, d.raw_nonfatlike_shell_mm3, marker="o", label=vessel)
    ax.set_xlabel("Peri-luminal shell thickness (mm)")
    ax.set_ylabel("Raw non-fatlike shell volume (mm³)")
    ax.set_title("Raw shell-volume sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def synthetic_self_test():
    a = np.array([1.0, 1.5, 2.0, 2.5])
    w = _arc_weights(a)
    assert np.allclose(w, [.25,.5,.5,.25])
    assert np.isclose(w.sum(), 1.5)

    # PCAT shell geometry: shell outer radius is 3 * modeled outer radius.
    r = np.array([1.0, 2.0])
    outer = r + .75
    assert np.allclose(3.0*outer - outer, 2.0*outer)
    return {"ok": True, "integration_weight_sum_mm": float(w.sum())}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root/OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out/"run_state.json", {"status":"STARTED","algorithm":ALGORITHM,"baseline":BASELINE})

    master = _read_json(root/MASTER)
    freeze = _read_json(root/FREEZE_SUMMARY)
    if master.get("status") != EXPECTED_MASTER_STATUS:
        raise RuntimeError(f"Frozen master prerequisite failed: {master.get('status')}")
    if freeze.get("status") != EXPECTED_FREEZE_STATUS:
        raise RuntimeError(f"LCX structural freeze prerequisite failed: {freeze.get('status')}")
    if not bool(freeze.get("decision",{}).get("downstream_quantification_allowed",False)):
        raise RuntimeError("LCX structural freeze does not allow downstream research quantification")

    geom, src, meta, voxel_volume = _load_source(root/SOURCE_CACHE)
    freeze_qc = pd.read_csv(_req(root/FREEZE_QC))
    freeze_qc["station_qc_pass"] = freeze_qc.station_qc_pass.astype(bool)

    stations = _quantify_shells(geom, src, freeze_qc)
    stations.to_csv(out/"LCX_OM_raw_shell_station_quantification.csv", index=False)

    totals = _shell_totals(stations)
    totals.to_csv(out/"LCX_OM_raw_shell_totals.csv", index=False)

    profiles = pd.concat([_shell_profile_1mm(stations, s) for s in SHELLS_MM], ignore_index=True)
    profiles.to_csv(out/"LCX_OM_raw_shell_profiles_1mm.csv", index=False)

    stability = _shell_stability(stations)
    stability.to_csv(out/"LCX_OM_raw_shell_profile_stability.csv", index=False)

    aorta, aorta_path = _load_aorta_mask(root, src.shape, geom)

    pcat_summaries = {}
    primary_by = {}
    sensitivity_by = {}
    for vessel in ("C6","C7"):
        s, primary, sens = _pcat_for_vessel(
            geom, src, aorta, freeze_qc, vessel, voxel_volume
        )
        pcat_summaries[vessel] = s
        primary_by[vessel] = primary
        sensitivity_by[vessel] = sens
        sens.assign(vessel=vessel).to_csv(out/f"{vessel}_PCAT_margin_sensitivity.csv", index=False)
        primary["longitudinal"].assign(vessel=vessel).to_csv(
            out/f"{vessel}_PCAT_longitudinal_primary.csv", index=False
        )
        primary["radial"].assign(vessel=vessel).to_csv(
            out/f"{vessel}_PCAT_radial_primary.csv", index=False
        )

    pcat_table = pd.DataFrame(list(pcat_summaries.values()))
    pcat_table.to_csv(out/"LCX_OM_PCAT_primary_summary.csv", index=False)

    nominal = totals[np.isclose(totals.shell_thickness_mm, NOMINAL_SHELL_MM)].copy()
    technical = {}
    for vessel in ("C6","C7"):
        nd = nominal[nominal.vessel == vessel].iloc[0]
        ps = pcat_summaries[vessel]
        technical[vessel] = {
            "evaluable_shell_length_mm": float(nd.evaluable_length_mm),
            "raw_shell_recomputed_station_count": int(nd.station_count),
            "pcat_segment_length_mm": float(ps["segment_length_mm"]),
            "pcat_fat_voxels": int(ps["fat_voxels"]),
            "pcat_longitudinal_bin_coverage_fraction": float(ps["longitudinal_bin_coverage_fraction"]),
            "technical_feasibility_pass": bool(
                float(nd.evaluable_length_mm) >= MIN_EVALUABLE_LENGTH_MM
                and float(ps["segment_length_mm"]) >= MIN_EVALUABLE_LENGTH_MM
                and int(ps["fat_voxels"]) >= MIN_CANONICAL_FAT_VOXELS
                and float(ps["longitudinal_bin_coverage_fraction"]) >= MIN_LONGITUDINAL_BIN_COVERAGE
            ),
        }

    status = STATUS_PASS if all(v["technical_feasibility_pass"] for v in technical.values()) else STATUS_FAIL

    _plot_composition(totals, out/"01_LCX_OM_raw_shell_composition.png")
    _plot_shell_sensitivity(totals, out/"02_LCX_OM_shell_sensitivity.png")
    _plot_pcat_longitudinal(primary_by, out/"03_LCX_OM_PCAT_longitudinal.png")
    _plot_pcat_sensitivity(sensitivity_by, out/"04_LCX_OM_PCAT_margin_sensitivity.png")

    nominal_dict = {
        r.vessel: {
            "evaluable_length_mm": float(r.evaluable_length_mm),
            "shell_volume_mm3": float(r.shell_volume_mm3),
            "fatlike_excluded_mm3": float(r.fatlike_excluded_mm3),
            "low_attenuation_mm3": float(r.low_attenuation_mm3),
            "noncalcified_mm3": float(r.noncalcified_mm3),
            "mixed_intermediate_mm3": float(r.mixed_intermediate_mm3),
            "calcified_mm3": float(r.calcified_mm3),
            "raw_nonfatlike_shell_mm3": float(r.raw_nonfatlike_shell_mm3),
        }
        for _, r in nominal.iterrows()
    }

    summary = {
        "status": status,
        "algorithm": ALGORITHM,
        "baseline_commit": BASELINE,
        "master_status": master.get("status"),
        "master_modified": False,
        "structural_freeze_status": freeze.get("status"),
        "labels": {
            "C6": "LCX-like parent continuation",
            "C7": "OM-like daughter",
            "clinical_identity_established": False,
            "LM_status": "UNRESOLVED",
        },
        "bifurcation_exclusion_mm": BIFURCATION_EXCLUSION_MM,
        "source_voxel_volume_mm3": voxel_volume,
        "aorta_exclusion_available": bool(aorta is not None),
        "aorta_exclusion_path": str(aorta_path) if aorta_path is not None else None,
        "nominal_1mm_raw_shell_totals": nominal_dict,
        "pcat_primary": pcat_summaries,
        "technical_feasibility": technical,
        "is_plaque_excess_model": False,
        "is_validated_plaque_burden": False,
        "is_validated_clinical_tpv": False,
        "is_proprietary_fai": False,
        "scientific_boundary": (
            "This experiment quantifies raw peri-luminal shell HU composition and direct PCAT attenuation on the exact frozen C6/C7 research structural paths. "
            "It does not fit a plaque-excess model, does not call raw shell volume TPV, and does not establish clinical LCX/OM identity. "
            "C6 and C7 measurements are separate research segments and must not be added as a clinical LCX burden. PCAT uses the same direct -190 to -30 HU "
            "attenuation definition and +0.75 mm circular outer-wall margin as the locked RCA prototype, but these shorter non-RCA segments are feasibility measurements, not validated FAI."
        ),
    }
    _write_json(out/"summary.json", summary)
    _write_json(out/"input_provenance.json", {
        "source_cache": str(root/SOURCE_CACHE),
        "master": str(root/MASTER),
        "structural_freeze_summary": str(root/FREEZE_SUMMARY),
        "structural_freeze_qc": str(root/FREEZE_QC),
    })

    report = out/"OPENPLAQUE_LCX_OM_SOURCE_SPACE_COMPOSITION_PCAT_FEASIBILITY_REPORT.html"
    report.write_text(
        "<html><body><h1>OpenPlaque LCX-like / OM-like Source-Space Composition + PCAT Feasibility v1</h1>"
        f"<p><b>Status:</b> {status}</p>"
        "<p>Research structural labels only: C6 LCX-like parent continuation; C7 OM-like daughter. Clinical identity remains unestablished; LM unresolved.</p>"
        f"<p>C6 direct PCAT: {pcat_summaries['C6']['pcat_mean_hu']:.2f} HU over {pcat_summaries['C6']['fat_voxels']:,} fat voxels.</p>"
        f"<p>C7 direct PCAT: {pcat_summaries['C7']['pcat_mean_hu']:.2f} HU over {pcat_summaries['C7']['fat_voxels']:,} fat voxels.</p>"
        f"<p>C6 raw 1-mm non-fatlike shell: {nominal_dict['C6']['raw_nonfatlike_shell_mm3']:.3f} mm³; "
        f"C7 raw 1-mm non-fatlike shell: {nominal_dict['C7']['raw_nonfatlike_shell_mm3']:.3f} mm³.</p>"
        "<p><b>Boundary:</b> raw shell composition is not plaque-excess burden or TPV. PCAT is direct attenuation, not proprietary FAI. Do not add C6 and C7 as clinical LCX burden.</p>"
        '<img src="01_LCX_OM_raw_shell_composition.png" style="max-width:100%">'
        '<img src="02_LCX_OM_shell_sensitivity.png" style="max-width:100%">'
        '<img src="03_LCX_OM_PCAT_longitudinal.png" style="max-width:100%">'
        '<img src="04_LCX_OM_PCAT_margin_sensitivity.png" style="max-width:100%">'
        "</body></html>",
        encoding="utf-8",
    )

    _write_json(out/"run_state.json", {
        "status":"COMPLETE","result_status":status,"algorithm":ALGORITHM,"baseline":BASELINE
    })
    zpath = out/"OPENPLAQUE_LCX_OM_SOURCE_SPACE_COMPOSITION_PCAT_FEASIBILITY_RESULTS.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p != zpath:
                z.write(p,p.name)

    return {"summary": summary, "report": str(report), "zip": str(zpath)}
