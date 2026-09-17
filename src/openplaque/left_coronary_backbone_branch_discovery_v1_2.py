from __future__ import annotations

"""Technical v1.2 robustness fix for left-coronary backbone branch discovery.

v1.1 corrected source/vesselness coordinates and short-path handling, but the known C7
positive control still failed. The failure is geometric: the nominal control point on the
label-neutral backbone is about 0.8 mm from the C7 reference split, mostly lateral to the
backbone. A single centerline seed is therefore too brittle for a branch-origin control.

This version changes only seed placement. Around every requested backbone station (including
blind stations and the C7 control), it evaluates a small source-space neighborhood: arc offsets
of +/-0.4 mm and a 0.8-mm cross-sectional ring plus the center. Preview directions from these
nearby starts are ranked globally, but the number of expensive beam traces remains unchanged.
All prospective HU, vesselness, branch-angle, endpoint-separation, dense-QC, control, and
acceptance gates are unchanged.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import left_coronary_backbone_branch_discovery_v1 as base
from . import left_coronary_backbone_branch_discovery_v1_1 as fix1

ALGORITHM = "left-coronary-backbone-branch-discovery-v1.2-cross-sectional-seeds"
SEED_ARC_OFFSETS_MM = (-0.40, 0.0, 0.40)
SEED_RADIAL_OFFSET_MM = 0.80
PREVIEW_DIRS_PER_START = 2


def _seed_start_candidates(backbone, seed_arc):
    """Generate a small deterministic neighborhood around one backbone seed station."""
    backbone = np.asarray(backbone, float)
    total = float(base._arc(backbone)[-1])
    ring = [
        (0.0, 0.0),
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (2 ** -0.5, 2 ** -0.5), (2 ** -0.5, -(2 ** -0.5)),
        (-(2 ** -0.5), 2 ** -0.5), (-(2 ** -0.5), -(2 ** -0.5)),
    ]
    out = []
    seen = set()
    for da in SEED_ARC_OFFSETS_MM:
        a = float(np.clip(float(seed_arc) + da, 0.0, total))
        c = base._interp(backbone, [a])[0]
        t = base._local_tangent(backbone, a)
        u, v = base._orth_basis(t)
        for ou, ov in ring:
            radial = float(SEED_RADIAL_OFFSET_MM * np.hypot(ou, ov))
            p = c + SEED_RADIAL_OFFSET_MM * (ou * u + ov * v)
            key = tuple(np.round(p, 4))
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "start": p,
                "tangent": t,
                "actual_seed_arc_mm": a,
                "arc_adjust_mm": a - float(seed_arc),
                "cross_u_mm": float(SEED_RADIAL_OFFSET_MM * ou),
                "cross_v_mm": float(SEED_RADIAL_OFFSET_MM * ov),
                "radial_offset_mm": radial,
            })
    return out


def _select_preview_rays(rays, n):
    """Keep high-scoring rays while guaranteeing start-point diversity first."""
    if not rays:
        return []
    rays = sorted(rays, key=lambda r: r["preview_score"], reverse=True)
    selected = []
    used_starts = set()

    # First pass: at most one ray per start point so a single bright center does not monopolize
    # the expensive beam budget.
    for r in rays:
        sk = tuple(np.round(r["start"], 3))
        if sk in used_starts:
            continue
        selected.append(r)
        used_starts.add(sk)
        if len(selected) >= min(int(n), 4):
            break

    # Second pass: fill the remaining beam slots by score, avoiding near-duplicate rays from
    # essentially the same start and direction.
    for r in rays:
        if len(selected) >= int(n):
            break
        duplicate = False
        for k in selected:
            start_close = float(np.linalg.norm(r["start"] - k["start"])) < 0.35
            dir_close = base._angle(r["direction"], k["direction"]) < 12.0
            if start_close and dir_close:
                duplicate = True
                break
        if not duplicate:
            selected.append(r)
    return selected[: int(n)]


def _evaluate_seed_multistart(ref, src, ves, backbone, backbone_tree, seed_arc, vthr, vnorm, tag):
    """v1.0 evaluator with robust local seed starts; all scientific gates are unchanged."""
    rays = []
    for sm in _seed_start_candidates(backbone, seed_arc):
        start = np.asarray(sm["start"], float)
        tan = np.asarray(sm["tangent"], float)
        previews = base._preview_dirs(
            ref, src, ves, start, tan, backbone_tree, vnorm, n=PREVIEW_DIRS_PER_START
        )
        start_hu = float(base._sample_array(ref, src, [start], cval=-1024.0)[0])
        start_vv = float(base._sample_array(ref, ves, [start], cval=0.0)[0])
        for preview, d in previews:
            rays.append({
                **sm,
                "start": start,
                "tangent": tan,
                "direction": np.asarray(d, float),
                "preview_score": float(preview),
                "start_source_hu": start_hu,
                "start_vesselness": start_vv,
            })

    chosen = _select_preview_rays(rays, base.INITIAL_DIRS)
    attrs, rows, paths, qcdfs = [], [], {}, {}

    for di, ray in enumerate(chosen):
        finals, attr = base._beam(
            ref, src, ves, ray["start"], ray["direction"], backbone_tree, vthr, vnorm
        )
        if len(attr):
            attr = attr.copy()
            attr["seed_arc_mm"] = float(seed_arc)
            attr["actual_seed_arc_mm"] = float(ray["actual_seed_arc_mm"])
            attr["arc_adjust_mm"] = float(ray["arc_adjust_mm"])
            attr["cross_u_mm"] = float(ray["cross_u_mm"])
            attr["cross_v_mm"] = float(ray["cross_v_mm"])
            attr["radial_offset_mm"] = float(ray["radial_offset_mm"])
            attr["dir_index"] = int(di)
            attr["preview_score"] = float(ray["preview_score"])
            attr["start_source_hu"] = float(ray["start_source_hu"])
            attr["start_vesselness"] = float(ray["start_vesselness"])
            attrs.append(attr)

        for fi, st in enumerate(finals[:2]):
            raw = np.asarray(st[1], float)
            p, q, qc = base._dense_qc(ref, src, raw)
            tr = base._truncate_qc(qc)
            hid = f"{tag}_d{di}_f{fi}"
            last = int(tr["accepted_last_index"])
            acc = p[: last + 1] if last >= 0 else p[:1]
            epsep = float(backbone_tree.query(acc[-1])[0])
            ang = base._branch_angle(acc, ray["tangent"])
            row = {
                "hypothesis_id": hid,
                "seed_arc_mm": float(seed_arc),
                "actual_seed_arc_mm": float(ray["actual_seed_arc_mm"]),
                "arc_adjust_mm": float(ray["arc_adjust_mm"]),
                "cross_u_mm": float(ray["cross_u_mm"]),
                "cross_v_mm": float(ray["cross_v_mm"]),
                "radial_offset_mm": float(ray["radial_offset_mm"]),
                "dir_index": int(di),
                "final_index": int(fi),
                "preview_score": float(ray["preview_score"]),
                "start_source_hu": float(ray["start_source_hu"]),
                "start_vesselness": float(ray["start_vesselness"]),
                "beam_score": float(st[0]),
                "path_arc_mm": float(q[-1]) if len(q) else 0.0,
                "branch_angle_deg": float(ang),
                "endpoint_backbone_separation_mm": epsep,
                **tr,
            }
            row["branch_gate_pass"] = bool(
                tr["accepted"]
                and ang >= base.MIN_BRANCH_ANGLE_DEG
                and epsep >= base.MIN_ENDPOINT_BACKBONE_SEPARATION_MM
            )
            rows.append(row)
            paths[hid] = acc
            qcdfs[hid] = qc

    return (
        pd.concat(attrs, ignore_index=True) if attrs else pd.DataFrame(),
        pd.DataFrame(rows),
        paths,
        qcdfs,
    )


def synthetic_multistart_seed_self_test():
    x = np.linspace(0.0, 20.0, 81)
    backbone = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    starts = _seed_start_candidates(backbone, 10.0)
    pts = np.asarray([s["start"] for s in starts])
    assert len(starts) >= 20
    assert any(abs(float(s["arc_adjust_mm"])) >= 0.39 for s in starts)
    radial = np.asarray([s["radial_offset_mm"] for s in starts])
    assert np.max(radial) >= 0.79
    # A target 0.8 mm off a straight backbone cross-section must be directly represented.
    target = np.array([10.0, 0.8, 0.0])
    assert float(np.min(np.linalg.norm(pts - target, axis=1))) < 0.05
    return {"ok": True, "candidate_starts": len(starts), "max_radial_offset_mm": float(np.max(radial))}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    # v1.1 still supplies the source-space vesselness and short-path technical fixes.
    # This patch changes only robust seed placement; all prospective scientific gates remain base v1.0.
    base._evaluate_seed = _evaluate_seed_multistart
    previous_algorithm = fix1.ALGORITHM
    fix1.ALGORITHM = ALGORITHM
    try:
        result = fix1.run(drive_root=drive_root, output_dir=output_dir)
    finally:
        fix1.ALGORITHM = previous_algorithm

    out = Path(output_dir) if output_dir else Path(drive_root) / base.OUTPUT_DIRNAME
    provenance = {
        "algorithm": ALGORITHM,
        "seed_arc_offsets_mm": list(SEED_ARC_OFFSETS_MM),
        "seed_radial_offset_mm": SEED_RADIAL_OFFSET_MM,
        "preview_dirs_per_start": PREVIEW_DIRS_PER_START,
        "beam_traces_per_requested_station": int(base.INITIAL_DIRS),
        "reason": "v1.1 C7 control seed was offset from the C7 reference split; robust local seeding prevents a sub-millimeter centerline discrepancy from invalidating the positive control",
        "scientific_change": "seed placement robustness only; HU/vesselness/branch-angle/endpoint-separation/dense-QC/control acceptance gates unchanged",
    }
    (out / "cross_section_seed_fix_v1_2.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return result
