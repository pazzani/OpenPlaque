from __future__ import annotations

"""Performance-safe implementation of blind source-CCTA coronary ostium discovery.

Scientific discovery weights, thresholds, candidate limits, RCA control logic, and final
acceptance gates remain those of v1.1. This version changes execution only:

1. The 3-D Frangi root vesselness calculation is resumably cached after each scale.
2. Serial orthogonal-plane QC is skipped for beam finalists that already fail one of the
   inexpensive non-QC gates and therefore cannot possibly be accepted.
3. Progress is written to progress.json and printed so Colab does not appear frozen.

A positive result still only nominates a second coronary-sized ostial exit for visual QC.
It does not establish clinical left-main identity or modify the frozen master baseline.
"""

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import left_coronary_source_ostium_discovery as _base
from . import left_coronary_source_ostium_discovery_v1_1 as _v11

BASELINE = _base.BASELINE
ALGORITHM = "left-coronary-source-ostium-discovery-v1.2-performance"
OUTPUT_DIRNAME = "Left_Coronary_Source_Ostium_Discovery_v1_2_fast"
CACHE_DIRNAME = "Cache/Left_Coronary_Source_Ostium_Discovery_v1_2"

_score_plane = _base._score_plane
synthetic_root_component_self_test = _base.synthetic_root_component_self_test
_component_index_and_geometry_points = _v11._component_index_and_geometry_points

_ACTIVE_PROGRESS_PATH: Path | None = None
_ACTIVE_START = 0.0
_ACTIVE_COMPONENT_TOTAL = 0
_ACTIVE_COMPONENT_INDEX = 0


def _jsonable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _progress(stage: str, **extra):
    payload = {
        "algorithm": ALGORITHM,
        "stage": stage,
        "elapsed_seconds": float(time.time() - _ACTIVE_START) if _ACTIVE_START else 0.0,
        **extra,
    }
    print("[v1.2]", stage, *(f"{k}={v}" for k, v in extra.items()))
    if _ACTIVE_PROGRESS_PATH is not None:
        _ACTIVE_PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _ACTIVE_PROGRESS_PATH.write_text(
            json.dumps(payload, indent=2, default=_jsonable, allow_nan=True), encoding="utf-8"
        )


def _source_digest(vol) -> str:
    a = np.asarray(vol)
    # Exact content identity. The root crop is modest compared with the Frangi calculation.
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def _frangi_cache_meta(vol, spacing, scales):
    return {
        "shape": list(np.asarray(vol).shape),
        "spacing": [float(x) for x in np.asarray(spacing, float)],
        "scales_mm": [float(x) for x in scales],
        "source_sha256": _source_digest(vol),
        "formula": "exact-v1.0-frangi",
    }


def _meta_matches(a, b):
    return (
        a.get("shape") == b.get("shape")
        and np.allclose(a.get("spacing", []), b.get("spacing", []), rtol=0, atol=1e-8)
        and np.allclose(a.get("scales_mm", []), b.get("scales_mm", []), rtol=0, atol=1e-8)
        and a.get("source_sha256") == b.get("source_sha256")
        and a.get("formula") == b.get("formula")
    )


def _frangi_one_scale(x, spacing, sm):
    """Exact single-scale computation copied from the frozen v1.0 Frangi formula."""
    spacing = np.asarray(spacing, float)
    sig = np.maximum(float(sm) / spacing, .55)
    n = float(sm) * float(sm)
    hzz = ndi.gaussian_filter(x, sig, order=(2, 0, 0), mode="nearest") * n / spacing[0] ** 2
    hyy = ndi.gaussian_filter(x, sig, order=(0, 2, 0), mode="nearest") * n / spacing[1] ** 2
    hxx = ndi.gaussian_filter(x, sig, order=(0, 0, 2), mode="nearest") * n / spacing[2] ** 2
    hzy = ndi.gaussian_filter(x, sig, order=(1, 1, 0), mode="nearest") * n / (spacing[0] * spacing[1])
    hzx = ndi.gaussian_filter(x, sig, order=(1, 0, 1), mode="nearest") * n / (spacing[0] * spacing[2])
    hyx = ndi.gaussian_filter(x, sig, order=(0, 1, 1), mode="nearest") * n / (spacing[1] * spacing[2])
    H = np.empty(x.shape + (3, 3), np.float32)
    H[..., 0, 0] = hzz
    H[..., 1, 1] = hyy
    H[..., 2, 2] = hxx
    H[..., 0, 1] = H[..., 1, 0] = hzy
    H[..., 0, 2] = H[..., 2, 0] = hzx
    H[..., 1, 2] = H[..., 2, 1] = hyx
    vals = np.linalg.eigvalsh(H)
    vals = np.take_along_axis(vals, np.argsort(np.abs(vals), axis=-1), axis=-1)
    l1, l2, l3 = vals[..., 0], vals[..., 1], vals[..., 2]
    eps = 1e-8
    ra = np.abs(l2) / (np.abs(l3) + eps)
    rb = np.abs(l1) / np.sqrt(np.abs(l2 * l3) + eps)
    ss = np.sqrt(l1 * l1 + l2 * l2 + l3 * l3)
    nz = ss[ss > 0]
    cc = max(float(np.percentile(nz, 90)) * .45 if nz.size else .05, 1e-4)
    v = (1 - np.exp(-(ra * ra) / .5)) * np.exp(-(rb * rb) / .5) * (
        1 - np.exp(-(ss * ss) / (2 * cc * cc))
    )
    v[(l2 >= 0) | (l3 >= 0)] = 0
    return np.nan_to_num(v).astype(np.float32)


def _frangi_3d_cached(vol, spacing, scales=(.60, .90, 1.25), cache_dir=None, reuse_cache=True):
    """Exact v1.0 Frangi calculation with resumable per-scale Drive caching."""
    scales = tuple(float(x) for x in scales)
    current = _frangi_cache_meta(vol, spacing, scales)
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    meta_path = cache_dir / "root_vesselness_meta.json" if cache_dir else None
    final_path = cache_dir / "root_vesselness.npy" if cache_dir else None
    partial_path = cache_dir / "root_vesselness_partial.npy" if cache_dir else None

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    if reuse_cache and final_path is not None and final_path.exists() and meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if _meta_matches(old, current) and old.get("completed_scales") == len(scales):
            _progress("vesselness_cache_hit", completed_scales=len(scales))
            return np.load(final_path, mmap_mode="r")

    x = np.clip(np.asarray(vol, np.float32), 80.0, 1000.0)
    x = (x - 80.0) / 920.0
    best = np.zeros_like(x, np.float32)
    start_i = 0

    if reuse_cache and partial_path is not None and partial_path.exists() and meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        k = int(old.get("completed_scales", 0))
        if _meta_matches(old, current) and 0 < k < len(scales):
            cached = np.load(partial_path, mmap_mode="r")
            if tuple(cached.shape) == tuple(best.shape):
                best[...] = cached
                start_i = k
                _progress("vesselness_partial_cache_hit", completed_scales=k, total_scales=len(scales))

    for i in range(start_i, len(scales)):
        sm = scales[i]
        _progress("vesselness_scale_start", scale_index=i + 1, total_scales=len(scales), scale_mm=sm)
        v = _frangi_one_scale(x, spacing, sm)
        best = np.maximum(best, v).astype(np.float32, copy=False)
        if cache_dir:
            np.save(partial_path, best)
            meta = {**current, "completed_scales": i + 1, "complete": False}
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _progress("vesselness_scale_complete", scale_index=i + 1, total_scales=len(scales), scale_mm=sm)

    if cache_dir:
        np.save(final_path, best)
        meta = {**current, "completed_scales": len(scales), "complete": True}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if partial_path.exists():
            partial_path.unlink()
    _progress("vesselness_complete", completed_scales=len(scales))
    return best


def _cheap_path_gate(m, v_thr):
    """The non-serial-QC portion of the frozen v1.1 acceptance gate."""
    return bool(
        m["length_mm"] >= 5.0
        and m["tortuosity"] <= 1.8
        and m["robust_hu_fraction"] >= .90
        and m["p10_vesselness"] >= .50 * v_thr
        and m["outside_aorta_gain_mm"] >= 3.0
    )


def _serial_gate(qsum, rca_cal):
    return bool(
        qsum["plane_pass_fraction"] >= .60
        and qsum["median_plane_score"] >= .60
        and .55 * rca_cal["median_radius_mm"]
        <= qsum.get("median_radius_mm", np.nan)
        <= min(3.4, 2.0 * rca_cal["median_radius_mm"] + .2)
    )


def _selection_score(m, qsum, v_thr):
    return float(
        .42 * qsum["median_plane_score"]
        + .30 * qsum["plane_pass_fraction"]
        + .12 * min(1, m["outside_aorta_gain_mm"] / 5)
        + .10 * min(1, m["median_vesselness"] / max(v_thr, 1e-6))
        + .06 * min(1, m["length_mm"] / 8)
    )


def _trace_component(comp, src_roi, vessel, outside, spacing, v_thr, rca_cal, label):
    """v1.1 tracing with logically safe pre-QC pruning of impossible-to-accept finalists."""
    global _ACTIVE_COMPONENT_INDEX
    _ACTIVE_COMPONENT_INDEX += 1
    _progress(
        "trace_component_start",
        component_index=_ACTIVE_COMPONENT_INDEX,
        component_total=_ACTIVE_COMPONENT_TOTAL,
        component_id=comp.get("component_id"),
    )

    _, pts, _, seed = _component_index_and_geometry_points(comp, outside)
    sp = np.asarray(spacing, float)
    near = pts[np.linalg.norm((pts - seed[None, :]) * sp[None, :], axis=1) <= 6.0]
    tangent = _base._component_direction(near, seed, spacing, outside)
    finals = _base._beam_source(seed, tangent, src_roi, vessel, outside, spacing, v_thr)

    evaluated = []
    rejected_cheap = []
    for st in finals:
        p = np.asarray(st[1])
        m = _base._path_metrics(p, src_roi, vessel, outside, spacing, v_thr)
        if not _cheap_path_gate(m, v_thr):
            rejected_cheap.append((float(st[0]), m, p))
            continue

        qdf, qsum = _base._serial_qc(p, src_roi, spacing, rca_cal, n=9, label=label)
        gate = bool(_serial_gate(qsum, rca_cal))
        score = _selection_score(m, qsum, v_thr)
        evaluated.append((gate, score, m, qsum, p, qdf, float(st[0])))

    # Any accepted candidate outranks every rejected one in the frozen sorting rule.
    accepted = [r for r in evaluated if r[0]]
    if accepted:
        accepted.sort(key=lambda r: r[1], reverse=True)
        gate, score, m, qsum, p, qdf, bscore = accepted[0]
    elif evaluated:
        evaluated.sort(key=lambda r: r[1], reverse=True)
        gate, score, m, qsum, p, qdf, bscore = evaluated[0]
    elif rejected_cheap:
        # No path can satisfy the frozen acceptance gate. Preserve a representative traced
        # path for post-hoc diagnostics without spending serial-QC time on impossible paths.
        rejected_cheap.sort(key=lambda r: r[0], reverse=True)
        bscore, m, p = rejected_cheap[0]
        gate = False
        score = float("nan")
        qsum = {
            "label": label,
            "median_plane_score": float("nan"),
            "plane_pass_fraction": float("nan"),
            "median_radius_mm": float("nan"),
            "serial_qc_skipped": True,
            "skip_reason": "failed_non_qc_acceptance_gate",
        }
        qdf = pd.DataFrame()
    else:
        _progress(
            "trace_component_complete",
            component_id=comp.get("component_id"),
            beam_finalists=0,
            pre_qc_pass=0,
            serial_qc_evaluated=0,
            accepted=False,
        )
        return None, {
            "accepted": False,
            "reason": "no_path_ge_5mm",
            "component_id": comp["component_id"],
            "seed_zyx_local": seed.tolist(),
            "performance": {
                "beam_finalists": 0,
                "pre_qc_pass": 0,
                "serial_qc_evaluated": 0,
                "serial_qc_skipped": 0,
            },
        }, pd.DataFrame()

    sm = {
        "accepted": bool(gate),
        "component_id": comp["component_id"],
        "component_n_voxels": comp["n_voxels"],
        "component_span_mm": comp["physical_span_mm"],
        "seed_zyx_local": seed.tolist(),
        "initial_tangent_mm": tangent.tolist(),
        "selection_score": float(score),
        "beam_score": float(bscore),
        "path_metrics": {k: v for k, v in m.items() if not isinstance(v, np.ndarray)},
        "serial_qc": qsum,
        "performance": {
            "beam_finalists": len(finals),
            "pre_qc_pass": len(evaluated),
            "serial_qc_evaluated": len(evaluated),
            "serial_qc_skipped": len(rejected_cheap),
        },
    }
    _progress(
        "trace_component_complete",
        component_id=comp.get("component_id"),
        beam_finalists=len(finals),
        pre_qc_pass=len(evaluated),
        serial_qc_evaluated=len(evaluated),
        serial_qc_skipped=len(rejected_cheap),
        accepted=bool(gate),
    )
    return p, sm, qdf


def run(
    drive_root="/content/drive/MyDrive/OpenPlaque",
    output_dir=None,
    reuse_vesselness_cache=True,
):
    """Run frozen v1.1 science with resumable vesselness and safe QC pre-pruning."""
    global _ACTIVE_PROGRESS_PATH, _ACTIVE_START, _ACTIVE_COMPONENT_TOTAL, _ACTIVE_COMPONENT_INDEX

    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = root / CACHE_DIRNAME
    _ACTIVE_PROGRESS_PATH = out / "progress.json"
    _ACTIVE_START = time.time()
    _ACTIVE_COMPONENT_TOTAL = 0
    _ACTIVE_COMPONENT_INDEX = 0
    _progress("started", reuse_vesselness_cache=bool(reuse_vesselness_cache), cache_dir=str(cache_dir))

    old_trace = _base._trace_component
    old_frangi = _base._frangi_3d
    old_blind = _base._blind_root_components
    old_algorithm = _base.ALGORITHM
    old_output = _base.OUTPUT_DIRNAME

    def frangi_wrapper(vol, spacing, scales=(.60, .90, 1.25)):
        return _frangi_3d_cached(
            vol,
            spacing,
            scales=scales,
            cache_dir=cache_dir,
            reuse_cache=reuse_vesselness_cache,
        )

    def blind_wrapper(src_roi, aorta_roi, vessel, spacing, v_thr, rca_local_z):
        global _ACTIVE_COMPONENT_TOTAL
        _progress("blind_component_detection_start")
        comps, outside = old_blind(src_roi, aorta_roi, vessel, spacing, v_thr, rca_local_z)
        _ACTIVE_COMPONENT_TOTAL = min(len(comps), 40)
        _progress(
            "blind_component_detection_complete",
            components_found=len(comps),
            components_to_trace=_ACTIVE_COMPONENT_TOTAL,
        )
        return comps, outside

    _base._trace_component = _trace_component
    _base._frangi_3d = frangi_wrapper
    _base._blind_root_components = blind_wrapper
    _base.ALGORITHM = ALGORITHM
    _base.OUTPUT_DIRNAME = OUTPUT_DIRNAME
    try:
        result = _base.run(drive_root, output_dir)
        _progress("complete", report=result.get("report"), zip=result.get("zip"))
        return result
    except Exception as exc:
        _progress("failed", exception_type=type(exc).__name__, exception=str(exc))
        raise
    finally:
        _base._trace_component = old_trace
        _base._frangi_3d = old_frangi
        _base._blind_root_components = old_blind
        _base.ALGORITHM = old_algorithm
        _base.OUTPUT_DIRNAME = old_output
