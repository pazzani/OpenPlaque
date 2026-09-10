from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .source_candidates import _slice_feature_map


@dataclass
class Aorta3DResult:
    mask: np.ndarray
    ascending_centers_zyx: np.ndarray
    root_slice_indices: np.ndarray
    threshold_hu: float
    work_spacing_mm: float


@dataclass
class RCA3DCandidate:
    z: int
    y: int
    x: int
    score: float
    hu: float
    vesselness: float
    local_radius_mm: float
    distance_from_aorta_mm: float
    support_slices: int
    aorta_y: float
    aorta_x: float


def _fit_shape(arr: np.ndarray, shape) -> np.ndarray:
    out = np.zeros(tuple(int(v) for v in shape), dtype=arr.dtype)
    src = tuple(slice(0, min(arr.shape[d], out.shape[d])) for d in range(3))
    dst = tuple(slice(0, min(arr.shape[d], out.shape[d])) for d in range(3))
    out[dst] = arr[src]
    return out


def segment_aorta_3d(volume_hu, image, *, threshold_hu=340.0, work_spacing_mm=1.0):
    """Identify the aortic contrast component using 3-D continuity through the arch.

    The volume is downsampled for robustness and memory efficiency. Candidate
    connected components are scored for simultaneous posterior descending-aorta
    occupancy and anterior ascending-aorta occupancy; this structurally rejects
    the pulmonary arterial tree better than a per-slice positional heuristic.
    """
    vol = np.asarray(volume_hu, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError("volume_hu must be 3-D")
    spacing_xyz = np.asarray(image.GetSpacing(), dtype=float)
    spacing_zyx = spacing_xyz[::-1]
    zoom = np.minimum(1.0, spacing_zyx / float(work_spacing_mm))
    work = ndi.zoom(vol, zoom=zoom, order=1, mode="nearest", prefilter=False)

    mask = work >= float(threshold_hu)
    z, h, w = mask.shape
    roi = np.zeros_like(mask, dtype=bool)
    roi[:, int(.08*h):int(.92*h), int(.08*w):int(.92*w)] = True
    mask &= roi

    labels, nlab = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if nlab == 0:
        raise RuntimeError("No bright 3-D components found")

    # Descending aorta: posterior and patient-left, which in a conventional
    # radiologic axial array is typically image-right of center. Ascending
    # aorta: anterior/central. Requiring one 3-D component to occupy both zones
    # exploits the arch connection and rejects the pulmonary tree.
    yy, xx = np.indices((h, w))
    posterior = (yy >= .56*h) & (yy <= .86*h) & (xx >= .43*w) & (xx <= .72*w)
    anterior = (yy >= .20*h) & (yy <= .62*h) & (xx >= .20*w) & (xx <= .62*w)

    counts = np.bincount(labels.ravel())
    best_lab = None
    best_score = -np.inf
    for lab in range(1, nlab + 1):
        count = int(counts[lab]) if lab < len(counts) else 0
        if count < 1000:
            continue
        zz, yv, xv = np.where(labels == lab)
        zspan = int(zz.max() - zz.min() + 1)
        if zspan < max(12, int(.10*z)):
            continue
        post_count = int(np.count_nonzero(posterior[yv, xv]))
        ant_count = int(np.count_nonzero(anterior[yv, xv]))
        if post_count == 0 or ant_count == 0:
            continue
        score = np.sqrt(float(post_count) * float(ant_count)) * np.log1p(zspan)
        if score > best_score:
            best_score = score
            best_lab = lab

    if best_lab is None:
        # Conservative fallback: prefer the large component with strongest
        # posterior-aorta occupancy and substantial z extent.
        for lab in range(1, nlab + 1):
            count = int(counts[lab]) if lab < len(counts) else 0
            if count < 1000:
                continue
            zz, yv, xv = np.where(labels == lab)
            zspan = int(zz.max() - zz.min() + 1)
            post_count = int(np.count_nonzero(posterior[yv, xv]))
            score = float(post_count) * np.log1p(max(zspan, 1))
            if score > best_score:
                best_score = score
                best_lab = lab
    if best_lab is None:
        raise RuntimeError("Could not identify aortic 3-D component")

    work_aorta = labels == best_lab
    # Upsample the component back to source-array geometry using nearest-neighbor.
    inv_zoom = 1.0 / zoom
    full = ndi.zoom(work_aorta.astype(np.uint8), zoom=inv_zoom, order=0, mode="nearest", prefilter=False).astype(bool)
    full = _fit_shape(full, vol.shape)

    # Track the anterior large cross-section of the aortic component on each
    # source slice. This is the ascending aorta; the descending aorta is posterior.
    centers = []
    for zi in range(vol.shape[0]):
        sl = full[zi]
        labs2, n2 = ndi.label(sl)
        best = None
        best_area = 0
        for lab in range(1, n2 + 1):
            yv, xv = np.where(labs2 == lab)
            area = len(yv)
            if area < 500:
                continue
            cy, cx = float(yv.mean()), float(xv.mean())
            # Ascending aorta is anterior relative to the descending thoracic aorta.
            if cy > .68*vol.shape[1]:
                continue
            if area > best_area:
                best_area = area
                best = (zi, cy, cx, area)
        if best is not None:
            centers.append(best)

    if len(centers) < 10:
        raise RuntimeError("Aortic component found, but ascending-aorta tracking was inadequate")
    centers = np.asarray(centers, dtype=float)

    # Coronary ostia are at the inferior end of the ascending aorta. Use DICOM
    # patient-Superior coordinate rather than assuming NumPy z direction.
    patient_z = []
    for zi, cy, cx, _ in centers:
        p = image.TransformContinuousIndexToPhysicalPoint((float(cx), float(cy), float(zi)))
        patient_z.append(float(p[2]))
    patient_z = np.asarray(patient_z)
    low = float(np.percentile(patient_z, 5))
    high = low + 38.0
    root_keep = (patient_z >= low) & (patient_z <= high)
    root_slices = centers[root_keep, 0].astype(int)
    if len(root_slices) < 8:
        order = np.argsort(patient_z)
        root_slices = centers[order[:min(80, len(order))], 0].astype(int)

    return Aorta3DResult(
        mask=full,
        ascending_centers_zyx=centers[:, :3],
        root_slice_indices=np.unique(root_slices),
        threshold_hu=float(threshold_hu),
        work_spacing_mm=float(work_spacing_mm),
    )


def _ascending_component_on_slice(aorta_mask, zi, center_yx):
    sl = np.asarray(aorta_mask[zi], dtype=bool)
    labs, n = ndi.label(sl)
    if n == 0:
        return None
    cy0, cx0 = center_yx
    best = None
    best_d = np.inf
    for lab in range(1, n + 1):
        yy, xx = np.where(labs == lab)
        if len(yy) < 300:
            continue
        cy, cx = float(yy.mean()), float(xx.mean())
        d = np.hypot(cy-cy0, cx-cx0)
        if d < best_d:
            best_d = d
            best = (labs == lab, cy, cx)
    return best


def find_rca_candidates_from_aorta3d(volume_hu, image, aorta_result: Aorta3DResult, *, n=8):
    """Find proximal RCA candidates in a thin shell around the true 3-D aorta."""
    vol = np.asarray(volume_hu, dtype=np.float32)
    spacing_xyz = np.asarray(image.GetSpacing(), dtype=float)
    sy, sx = float(spacing_xyz[1]), float(spacing_xyz[0])
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)

    center_by_z = {int(round(z)): (float(y), float(x)) for z, y, x in aorta_result.ascending_centers_zyx}
    raw = []
    for zi in aorta_result.root_slice_indices:
        zi = int(zi)
        if zi not in center_by_z:
            continue
        comp = _ascending_component_on_slice(aorta_result.mask, zi, center_by_z[zi])
        if comp is None:
            continue
        aorta2d, cy, cx = comp
        score_map, vesselness, radius = _slice_feature_map(vol[zi], (sy, sx))

        # Distance outside the actual aortic lumen boundary.
        dist_out = ndi.distance_transform_edt(~aorta2d, sampling=(sy, sx))
        yy, xx = np.indices(vol.shape[1:])

        # Relative DICOM patient coordinates on the slice. DICOM +X is patient
        # left and +Y is posterior. RCA origin should lie on the patient-right
        # half of the aortic root and is usually not strongly posterior.
        dx_idx = (xx - cx) * sx
        dy_idx = (yy - cy) * sy
        d_patient_x = direction[0, 0] * dx_idx + direction[0, 1] * dy_idx
        d_patient_y = direction[1, 0] * dx_idx + direction[1, 1] * dy_idx

        shell = (dist_out >= 0.4) & (dist_out <= 10.0) & (~aorta2d)
        right = d_patient_x <= 3.0
        anteriorish = d_patient_y <= 10.0
        small = (radius >= 0.45) & (radius <= 3.2)
        bright = vol[zi] >= 180.0
        allowed = shell & right & anteriorish & small & bright

        wall_weight = np.exp(-0.5 * ((dist_out - 2.0) / 3.0) ** 2)
        side_weight = 1.0 / (1.0 + np.exp((d_patient_x - 0.5) / 2.0))
        local = score_map * wall_weight * (0.45 + 0.55*side_weight)
        local[~allowed] = 0.0
        if not np.any(local > 0):
            continue
        iy, ix = np.unravel_index(int(np.argmax(local)), local.shape)
        raw.append(dict(
            z=zi, y=int(iy), x=int(ix), base=float(local[iy, ix]),
            hu=float(vol[zi, iy, ix]), vesselness=float(vesselness[iy, ix]),
            radius=float(radius[iy, ix]), distance=float(dist_out[iy, ix]),
            aorta_y=cy, aorta_x=cx,
        ))

    if not raw:
        return []

    # Persistence across adjacent source slices strongly favors a real tubular
    # coronary over one-slice noise or calcific edges.
    for c in raw:
        support = 0
        for d in raw:
            dz = abs(d['z'] - c['z']) * float(spacing_xyz[2])
            dxy = np.hypot((d['x']-c['x'])*sx, (d['y']-c['y'])*sy)
            if dz <= 3.0 and dxy <= 4.0:
                support += 1
        c['support'] = support
        c['score'] = c['base'] * (1.0 + 0.10*min(support, 14))

    chosen = []
    for c in sorted(raw, key=lambda q: q['score'], reverse=True):
        duplicate = False
        for p in chosen:
            dz = abs(c['z']-p.z)*float(spacing_xyz[2])
            dxy = np.hypot((c['x']-p.x)*sx, (c['y']-p.y)*sy)
            if dz < 4.0 and dxy < 5.0:
                duplicate = True
                break
        if duplicate:
            continue
        chosen.append(RCA3DCandidate(
            z=c['z'], y=c['y'], x=c['x'], score=float(c['score']),
            hu=c['hu'], vesselness=c['vesselness'], local_radius_mm=c['radius'],
            distance_from_aorta_mm=c['distance'], support_slices=int(c['support']),
            aorta_y=c['aorta_y'], aorta_x=c['aorta_x'],
        ))
        if len(chosen) >= int(n):
            break
    return chosen
