from __future__ import annotations

import json
import math
import shutil
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "rca-expert-outer-wall-annotation-pack-v1.0"
OUTPUT_DIRNAME = "RCA_Expert_Outer_Wall_Annotation_Pack_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
RCA_CENTERLINE = Path("Cache/Source_Volume_Coronary_Centerlines/RCA_source_centerline.csv")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
LOCK_SUMMARY = Path("RCA_Plaque_PCAT_Research_Lock_v1/summary.json")

EXPECTED_LOCK_STATUS = "RCA_RESEARCH_PLAQUE_PCAT_BENCHMARK_LOCKED"
STATUS = "RCA_EXPERT_OUTER_WALL_ANNOTATION_PACK_READY"

ANNOTATION_ARC_START_MM = 0.0
ANNOTATION_ARC_END_MM = 50.0
ANNOTATION_STEP_MM = 1.0

PLANE_HALF_WIDTH_MM = 5.0
PLANE_PIXEL_MM = 0.15
N_ANGLES = 72
RADIAL_STEP_MM = 0.10
LUMEN_MAX_RADIUS_MM = 4.0
LUMEN_MIN_RADIUS_MM = 0.55

HU_BINS = {
    "fatlike_excluded": [-1e9, -30.0],
    "low_attenuation": [-30.0, 30.0],
    "noncalcified": [30.0, 130.0],
    "mixed_intermediate": [130.0, 350.0],
    "calcified": [350.0, 1e9],
}


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


def _cmedian(x):
    return np.median(np.stack([np.roll(x, k) for k in range(-2, 3)]), axis=0)


def _lumen_boundary(geom, src, c, t):
    u, v = _orth_basis(t)
    th = np.linspace(0, 2*np.pi, N_ANGLES, endpoint=False)
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
    valid_fraction = float(valid.mean())
    if valid.any():
        med = float(np.median(radii[valid]))
        radii[~valid] = med
        radii = _cmedian(radii)
        radii = np.clip(radii, med-.75, med+.75)
        radii = np.clip(radii, .55, 4.0)
        p10, p50, p90 = np.percentile(radii, [10, 50, 90])
        axis = float(p90 / max(p10, 1e-6))
    else:
        p10 = p50 = p90 = axis = np.nan

    qc = bool(center_hu >= 200 and valid_fraction >= .60 and np.isfinite(axis) and axis <= 2.5)
    return {
        "center_hu": center_hu,
        "lumen_threshold_hu": threshold,
        "valid_radial_fraction": valid_fraction,
        "lumen_radius_p10_mm": float(p10),
        "lumen_radius_median_mm": float(p50),
        "lumen_radius_p90_mm": float(p90),
        "lumen_axis_proxy": float(axis),
        "station_qc_pass": qc,
        "theta": th,
        "radii": radii,
        "u": u,
        "v": v,
    }


def _plane_grid(c, u, v, half=PLANE_HALF_WIDTH_MM, pixel=PLANE_PIXEL_MM):
    q = np.arange(-half, half + 1e-9, pixel)
    yy, xx = np.meshgrid(q, q, indexing="ij")
    pts = c + xx[..., None]*u + yy[..., None]*v
    return pts, q, xx, yy


def _lumen_mask_from_boundary(xx, yy, theta, radii):
    ang = np.mod(np.arctan2(yy, xx), 2*np.pi)
    r = np.sqrt(xx*xx + yy*yy)
    th = np.asarray(theta, float)
    rb = np.asarray(radii, float)
    th_ext = np.r_[th, 2*np.pi]
    rb_ext = np.r_[rb, rb[0]]
    rlim = np.interp(ang.ravel(), th_ext, rb_ext).reshape(ang.shape)
    return (r <= rlim).astype(np.uint8)


def _outer_contour_from_mask(mask):
    mask = np.asarray(mask, bool)
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    er = mask.copy()
    er[1:-1, 1:-1] &= mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    return mask & ~er


def _validate_outer_mask(hu_stack, lumen_stack, outer_stack, pixel_mm=PLANE_PIXEL_MM, step_mm=ANNOTATION_STEP_MM):
    hu_stack = np.asarray(hu_stack, float)
    lumen_stack = np.asarray(lumen_stack, bool)
    outer_stack = np.asarray(outer_stack, bool)

    if hu_stack.shape != lumen_stack.shape or hu_stack.shape != outer_stack.shape:
        raise ValueError("HU, lumen, and outer masks must have identical shapes")
    if hu_stack.ndim != 3:
        raise ValueError("Expected stack shape [slice, y, x]")

    rows = []
    pixel_area = float(pixel_mm*pixel_mm)
    voxel_vol = pixel_area*float(step_mm)
    for i in range(len(hu_stack)):
        lum = lumen_stack[i]
        out = outer_stack[i]
        contain = bool(np.all(out[lum]))
        wall = out & ~lum
        h = hu_stack[i]
        vals = h[wall]

        rec = {
            "slice_index": i,
            "lumen_pixels": int(lum.sum()),
            "outer_pixels": int(out.sum()),
            "wall_pixels": int(wall.sum()),
            "outer_contains_lumen": contain,
            "wall_area_mm2": float(wall.sum()*pixel_area),
            "wall_volume_mm3": float(wall.sum()*voxel_vol),
        }
        if len(vals):
            rec["wall_mean_hu"] = float(np.mean(vals))
            rec["fatlike_excluded_mm3"] = float(np.sum(vals < -30)*voxel_vol)
            rec["low_attenuation_mm3"] = float(np.sum((vals >= -30) & (vals < 30))*voxel_vol)
            rec["noncalcified_mm3"] = float(np.sum((vals >= 30) & (vals < 130))*voxel_vol)
            rec["mixed_intermediate_mm3"] = float(np.sum((vals >= 130) & (vals < 350))*voxel_vol)
            rec["calcified_mm3"] = float(np.sum(vals >= 350)*voxel_vol)
        else:
            rec.update({
                "wall_mean_hu": np.nan,
                "fatlike_excluded_mm3": 0.0,
                "low_attenuation_mm3": 0.0,
                "noncalcified_mm3": 0.0,
                "mixed_intermediate_mm3": 0.0,
                "calcified_mm3": 0.0,
            })
        rec["total_plaque_proxy_mm3"] = (
            rec["low_attenuation_mm3"] + rec["noncalcified_mm3"]
            + rec["mixed_intermediate_mm3"] + rec["calcified_mm3"]
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def _save_preview_png(path, hu, lumen_mask, plane_id, arc_mm):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    extent = [-PLANE_HALF_WIDTH_MM, PLANE_HALF_WIDTH_MM, PLANE_HALF_WIDTH_MM, -PLANE_HALF_WIDTH_MM]
    ax.imshow(hu, cmap="gray", vmin=-100, vmax=800, extent=extent)
    boundary = _outer_contour_from_mask(lumen_mask)
    yy, xx = np.nonzero(boundary)
    x = -PLANE_HALF_WIDTH_MM + xx*PLANE_PIXEL_MM
    y = -PLANE_HALF_WIDTH_MM + yy*PLANE_PIXEL_MM
    ax.scatter(x, y, s=2)
    ax.set_title(f"{plane_id} | source CCTA")
    ax.set_xlabel("plane x (mm)")
    ax.set_ylabel("plane y (mm)")
    ax.text(
        0.02, 0.02, f"arc {arc_mm:.1f} mm",
        transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=.7),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _make_montage(preview_dir, plane_ids, out):
    chosen = np.linspace(0, len(plane_ids)-1, min(12, len(plane_ids))).round().astype(int)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = np.asarray(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, chosen):
        img = plt.imread(preview_dir / f"{plane_ids[idx]}.png")
        ax.imshow(img)
        ax.axis("off")
    fig.suptitle("RCA expert outer-wall annotation pack — source-plane previews")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def synthetic_self_test():
    q = np.arange(-2.0, 2.0 + 1e-9, .1)
    yy, xx = np.meshgrid(q, q, indexing="ij")
    theta = np.linspace(0, 2*np.pi, 72, endpoint=False)
    radii = np.full(72, 1.0)
    lum = _lumen_mask_from_boundary(xx, yy, theta, radii)
    assert abs(lum.sum()*.1*.1 - np.pi) < .15

    hu = np.zeros((2, len(q), len(q)), float)
    lum3 = np.stack([lum, lum])
    outer = np.sqrt(xx*xx + yy*yy) <= 1.5
    out3 = np.stack([outer, outer])
    v = _validate_outer_mask(hu, lum3, out3, pixel_mm=.1, step_mm=1.0)
    assert v.outer_contains_lumen.all()
    assert (v.wall_volume_mm3 > 0).all()
    return {
        "ok": True,
        "synthetic_lumen_area_mm2": float(lum.sum()*.01),
        "synthetic_wall_volume_mm3": float(v.wall_volume_mm3.iloc[0]),
    }


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "run_state.json", {"status": "STARTED", "algorithm": ALGORITHM, "baseline": BASELINE})

    master = _read_json(root / MASTER)
    lock = _read_json(root / LOCK_SUMMARY)
    if master.get("status") != "CORONARY_ANATOMY_BASELINE_V2_FROZEN":
        raise RuntimeError("Frozen master prerequisite failed")
    if lock.get("status") != EXPECTED_LOCK_STATUS:
        raise RuntimeError(f"RCA research-lock prerequisite failed: {lock.get('status')}")

    geom, src, voxel_volume = _load_source(root / SOURCE_CACHE)
    rca = _load_path(root / RCA_CENTERLINE)
    full_arc = float(_arc(rca)[-1])
    if full_arc < ANNOTATION_ARC_END_MM:
        raise RuntimeError(f"RCA path only {full_arc:.2f} mm; need >= {ANNOTATION_ARC_END_MM:.1f} mm")

    arcs = np.arange(ANNOTATION_ARC_START_MM, ANNOTATION_ARC_END_MM + 1e-9, ANNOTATION_STEP_MM)
    centers = _interp(rca, arcs)
    tangents = np.gradient(centers, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)

    pack = out / "expert_pack"
    if pack.exists():
        shutil.rmtree(pack)
    previews = pack / "previews"
    previews.mkdir(parents=True, exist_ok=True)

    hu_stack = []
    lumen_stack = []
    manifest_rows = []
    u_stack = []
    v_stack = []
    radii_stack = []
    theta_stack = []
    plane_ids = []

    for i, (arc_mm, c, t) in enumerate(zip(arcs, centers, tangents)):
        L = _lumen_boundary(geom, src, c, t)
        if not L["station_qc_pass"]:
            raise RuntimeError(f"Lumen QC failed at expert-annotation arc {arc_mm:.1f} mm")
        pts, q, xx, yy = _plane_grid(c, L["u"], L["v"])
        hu = _sample(geom, src, pts.reshape(-1, 3)).reshape(len(q), len(q)).astype(np.float32)
        lum = _lumen_mask_from_boundary(xx, yy, L["theta"], L["radii"]).astype(np.uint8)

        plane_id = f"RCA_WALL_{i:03d}"
        plane_ids.append(plane_id)
        hu_stack.append(hu)
        lumen_stack.append(lum)
        u_stack.append(L["u"])
        v_stack.append(L["v"])
        radii_stack.append(L["radii"])
        theta_stack.append(L["theta"])

        _save_preview_png(previews / f"{plane_id}.png", hu, lum, plane_id, float(arc_mm))

        manifest_rows.append({
            "slice_index": i,
            "plane_id": plane_id,
            "arc_mm": float(arc_mm),
            "center_lps_x_mm": float(c[0]),
            "center_lps_y_mm": float(c[1]),
            "center_lps_z_mm": float(c[2]),
            "tangent_lps_x": float(t[0]),
            "tangent_lps_y": float(t[1]),
            "tangent_lps_z": float(t[2]),
            "plane_u_lps_x": float(L["u"][0]),
            "plane_u_lps_y": float(L["u"][1]),
            "plane_u_lps_z": float(L["u"][2]),
            "plane_v_lps_x": float(L["v"][0]),
            "plane_v_lps_y": float(L["v"][1]),
            "plane_v_lps_z": float(L["v"][2]),
            "center_hu": float(L["center_hu"]),
            "lumen_radius_median_mm": float(L["lumen_radius_median_mm"]),
            "lumen_axis_proxy": float(L["lumen_axis_proxy"]),
            "lumen_qc_pass": bool(L["station_qc_pass"]),
        })

    hu_stack = np.stack(hu_stack).astype(np.float32)
    lumen_stack = np.stack(lumen_stack).astype(np.uint8)
    outer_template = np.zeros_like(lumen_stack, dtype=np.uint8)
    u_stack = np.stack(u_stack).astype(np.float32)
    v_stack = np.stack(v_stack).astype(np.float32)
    radii_stack = np.stack(radii_stack).astype(np.float32)
    theta_stack = np.stack(theta_stack).astype(np.float32)

    np.save(pack / "RCA_source_HU_planes.npy", hu_stack)
    np.save(pack / "RCA_lumen_seed_mask.npy", lumen_stack)
    np.save(pack / "RCA_outer_wall_annotation_template.npy", outer_template)
    np.savez_compressed(
        pack / "RCA_expert_outer_wall_planes.npz",
        hu=hu_stack,
        lumen_mask=lumen_stack,
        outer_wall_template=outer_template,
        arc_mm=arcs.astype(np.float32),
        centers_lps_mm=centers.astype(np.float32),
        tangents_lps=tangents.astype(np.float32),
        plane_u_lps=u_stack,
        plane_v_lps=v_stack,
        lumen_radii_mm=radii_stack,
        lumen_theta_rad=theta_stack,
        plane_pixel_mm=np.array([PLANE_PIXEL_MM], np.float32),
        longitudinal_step_mm=np.array([ANNOTATION_STEP_MM], np.float32),
    )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(pack / "RCA_expert_outer_wall_manifest.csv", index=False)

    spec = {
        "status": "ANNOTATION_TEMPLATE_EMPTY",
        "mask_semantics": (
            "Expert output mask should be binary uint8 with value 1 for every pixel INSIDE the expert outer-vessel boundary, "
            "including the lumen. Wall tissue is computed later as outer_vessel_mask AND NOT lumen_seed_mask."
        ),
        "stack_shape_slice_y_x": list(hu_stack.shape),
        "slice_count": int(len(hu_stack)),
        "arc_range_mm": [float(arcs.min()), float(arcs.max())],
        "longitudinal_step_mm": ANNOTATION_STEP_MM,
        "plane_pixel_mm": PLANE_PIXEL_MM,
        "plane_half_width_mm": PLANE_HALF_WIDTH_MM,
        "source_values": "HU in float32 RCA_source_HU_planes.npy",
        "lumen_seed_mask": "RCA_lumen_seed_mask.npy; automated research lumen contour supplied as context",
        "outer_wall_template": "RCA_outer_wall_annotation_template.npy; replace with completed binary outer-vessel mask",
        "blinding": (
            "No plaque-vote labels, plaque-positive/negative labels, locked plaque values, or PCAT values are included in the expert pack."
        ),
        "hu_bins_for_later_validation": HU_BINS,
    }
    _write_json(pack / "annotation_spec.json", spec)

    instructions = """# RCA expert outer-wall annotation instructions

This package is intentionally blind to prior plaque-vote labels and PCAT results.

## What to annotate

For each cross-sectional source-CCTA plane, draw the outer vessel boundary of the RCA. The completed binary mask must contain value 1 for every pixel inside the outer vessel boundary, including the lumen. Do not draw only a ring.

The supplied RCA_lumen_seed_mask.npy is the automated research lumen boundary and may be used as anatomical context. The final expert outer-wall contour should be based on the source CCTA appearance, not on any fixed wall thickness.

## Files

- RCA_source_HU_planes.npy: float32 HU stack with shape [slice, y, x].
- RCA_lumen_seed_mask.npy: automated lumen seed mask.
- RCA_outer_wall_annotation_template.npy: all-zero mask with the required output shape.
- RCA_expert_outer_wall_manifest.csv: slice-to-arc/LPS geometry mapping.
- previews/*.png: windowed source-plane previews with the lumen seed outline only.
- annotation_spec.json: mask semantics and pixel spacing.

## Required expert output

Save the completed binary mask as RCA_outer_wall_expert_mask.npy with exactly the same shape as RCA_outer_wall_annotation_template.npy.

Pixel values must be 0 or 1. Every lumen pixel should also be inside the outer-vessel mask. If the outer wall is not interpretable on a slice, leave that entire slice zero rather than guessing.

## Blinding

Do not use the OpenPlaque plaque-vote profile, locked plaque excess result, or PCAT attenuation while contouring. Those results are intentionally omitted from this pack.
"""
    (pack / "README_EXPERT_ANNOTATION.md").write_text(instructions, encoding="utf-8")

    _make_montage(previews, plane_ids, out / "01_RCA_expert_annotation_pack_montage.png")

    template_validation = _validate_outer_mask(hu_stack, lumen_stack, outer_template)
    template_is_empty = bool((template_validation.outer_pixels == 0).all())

    report = out / "OPENPLAQUE_RCA_EXPERT_OUTER_WALL_ANNOTATION_PACK_REPORT.html"
    report.write_text(
        "<html><body><h1>OpenPlaque RCA Expert Outer-Wall Annotation Pack v1</h1>"
        f"<p><b>Status:</b> {STATUS}</p>"
        f"<p>Prepared {len(arcs)} blinded source-CCTA cross sections from {arcs.min():.1f} to {arcs.max():.1f} mm "
        f"at {ANNOTATION_STEP_MM:.1f}-mm longitudinal spacing and {PLANE_PIXEL_MM:.2f}-mm in-plane sampling.</p>"
        "<p>No plaque-vote labels, plaque benchmark values, or PCAT values are included in the expert pack.</p>"
        "<p>The supplied lumen mask is context only. Expert output must be a binary mask of the area inside the outer-vessel boundary.</p>"
        "<p><b>Purpose:</b> obtain an independent outer-wall reference before any further plaque-method development.</p>"
        '<img src="01_RCA_expert_annotation_pack_montage.png" style="max-width:100%">'
        "</body></html>",
        encoding="utf-8",
    )

    pack_zip = out / "OPENPLAQUE_RCA_EXPERT_OUTER_WALL_ANNOTATION_PACK.zip"
    with zipfile.ZipFile(pack_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in pack.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(pack))

    summary = {
        "status": STATUS,
        "algorithm": ALGORITHM,
        "baseline_commit": BASELINE,
        "master_status": master.get("status"),
        "master_modified": False,
        "rca_lock_status": lock.get("status"),
        "source_voxel_volume_mm3": voxel_volume,
        "annotation_slice_count": int(len(arcs)),
        "annotation_arc_start_mm": float(arcs.min()),
        "annotation_arc_end_mm": float(arcs.max()),
        "annotation_step_mm": ANNOTATION_STEP_MM,
        "plane_pixel_mm": PLANE_PIXEL_MM,
        "plane_shape_y_x": [int(hu_stack.shape[1]), int(hu_stack.shape[2])],
        "lumen_qc_fraction": float(manifest.lumen_qc_pass.mean()),
        "empty_template_confirmed": template_is_empty,
        "expert_pack_zip": str(pack_zip),
        "expected_expert_output": "RCA_outer_wall_expert_mask.npy",
        "next_validation": (
            "When a blinded expert outer-wall mask is available, compute true outer-wall-defined wall/plaque volumes from the same HU planes "
            "and compare the locked fixed-shell excess proxy against the expert-contoured reference without retuning the locked method."
        ),
        "scientific_boundary": (
            "This run creates a label-blind annotation package only. It does not modify the locked plaque benchmark, does not create expert labels, "
            "and does not claim clinical TPV."
        ),
    }
    _write_json(out / "summary.json", summary)
    _write_json(out / "input_provenance.json", {
        "source_cache": str(root / SOURCE_CACHE),
        "rca_centerline": str(root / RCA_CENTERLINE),
        "master": str(root / MASTER),
        "rca_lock_summary": str(root / LOCK_SUMMARY),
    })
    _write_json(out / "run_state.json", {
        "status": "COMPLETE",
        "result_status": STATUS,
        "algorithm": ALGORITHM,
        "baseline": BASELINE,
    })

    results_zip = out / "OPENPLAQUE_RCA_EXPERT_OUTER_WALL_ANNOTATION_PACK_RESULTS.zip"
    with zipfile.ZipFile(results_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p != results_zip:
                z.write(p, p.name)

    return {
        "summary": summary,
        "report": str(report),
        "expert_pack_zip": str(pack_zip),
        "results_zip": str(results_zip),
    }
