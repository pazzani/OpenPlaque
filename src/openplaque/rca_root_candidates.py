from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import frangi
from skimage.measure import regionprops


@dataclass
class AortaDetection:
    z: int
    y: float
    x: float
    radius_mm: float
    score: float
    label: int


@dataclass
class RootCandidate:
    z: int
    y: int
    x: int
    score: float
    hu: float
    vesselness: float
    local_radius_mm: float
    support_slices: int
    aorta_y: float
    aorta_x: float
    aorta_radius_mm: float


def _physical_xyz(image, z: float, y: float, x: float) -> np.ndarray:
    return np.asarray(
        image.TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z))),
        dtype=float,
    )


def _central_crop_bounds(shape_yx, frac_y=0.70, frac_x=0.70):
    h, w = int(shape_yx[0]), int(shape_yx[1])
    cy, cx = h / 2.0, w / 2.0
    hh, hw = h * float(frac_y) / 2.0, w * float(frac_x) / 2.0
    return (
        max(0, int(round(cy - hh))), min(h, int(round(cy + hh))),
        max(0, int(round(cx - hw))), min(w, int(round(cx + hw))),
    )


def _aorta_components(slice_hu, spacing_xy_mm, image, z, threshold_hu=260.0):
    """Return plausible large round blood-pool components for one axial slice."""
    sy, sx = float(spacing_xy_mm[1]), float(spacing_xy_mm[0])
    pix_area = sy * sx
    smooth = ndi.gaussian_filter(np.asarray(slice_hu, dtype=np.float32), sigma=1.0)
    mask = smooth >= float(threshold_hu)
    mask = ndi.binary_closing(mask, iterations=1)

    y0, y1, x0, x1 = _central_crop_bounds(mask.shape, 0.72, 0.72)
    work = np.zeros_like(mask, dtype=bool)
    work[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    labels, n = ndi.label(work)
    if n == 0:
        return [], labels

    center_phys = _physical_xyz(image, z, mask.shape[0] / 2.0, mask.shape[1] / 2.0)
    out = []
    for rp in regionprops(labels):
        area_mm2 = float(rp.area) * pix_area
        if area_mm2 < pi * 8.0**2 or area_mm2 > pi * 24.0**2:
            continue
        radius_mm = float(np.sqrt(area_mm2 / pi))
        y, x = rp.centroid
        phys = _physical_xyz(image, z, y, x)

        # DICOM patient coordinates are LPS: smaller X is patient-right and
        # smaller Y is anterior. This separates the ascending aorta from the
        # more patient-left pulmonary trunk in ordinary axial CCTA.
        dx_right = float(center_phys[0] - phys[0])
        dy_ant = float(center_phys[1] - phys[1])
        if dx_right < -8.0 or dy_ant < -20.0:
            continue

        ecc = float(getattr(rp, "eccentricity", 1.0))
        circular = np.exp(-2.2 * ecc**2)
        rscore = np.exp(-0.5 * ((radius_mm - 15.0) / 5.0) ** 2)
        xscore = np.exp(-0.5 * ((dx_right - 18.0) / 22.0) ** 2)
        yscore = np.exp(-0.5 * ((dy_ant - 22.0) / 28.0) ** 2)
        score = float(circular * rscore * xscore * yscore)
        out.append(AortaDetection(int(z), float(y), float(x), radius_mm, score, int(rp.label)))

    return sorted(out, key=lambda a: a.score, reverse=True), labels


def detect_ascending_aorta_track(volume_hu, image, *, z_fraction=(0.46, 0.76), threshold_hu=260.0):
    """Detect the patient-right ascending aorta across a cardiac root window."""
    vol = np.asarray(volume_hu, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError("volume_hu must be 3-D")
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    z0 = max(0, int(round(vol.shape[0] * float(z_fraction[0]))))
    z1 = min(vol.shape[0], int(round(vol.shape[0] * float(z_fraction[1]))))

    raw = []
    for z in range(z0, z1):
        cands, _ = _aorta_components(vol[z], spacing[:2], image, z, threshold_hu)
        raw.append(cands[:3])

    dp = []
    back = []
    for i, cands in enumerate(raw):
        if not cands:
            dp.append([]); back.append([]); continue
        vals = []
        prevs = []
        for c in cands:
            best = c.score
            best_j = -1
            for k in range(i - 1, max(-1, i - 5), -1):
                if k >= 0 and raw[k] and dp[k]:
                    for j, p in enumerate(raw[k]):
                        dpx = np.hypot(c.x - p.x, c.y - p.y)
                        penalty = 0.025 * dpx + 0.08 * abs(c.radius_mm - p.radius_mm)
                        v = dp[k][j] + c.score - penalty
                        if v > best:
                            best = v; best_j = (k, j)
                    break
            vals.append(float(best)); prevs.append(best_j)
        dp.append(vals); back.append(prevs)

    best_state = None
    best_val = -np.inf
    for i, vals in enumerate(dp):
        for j, v in enumerate(vals):
            if v > best_val:
                best_val = v; best_state = (i, j)
    if best_state is None:
        raise ValueError("Could not detect ascending aorta")

    chosen = {}
    state = best_state
    while state is not None:
        i, j = state
        c = raw[i][j]
        chosen[c.z] = c
        prev = back[i][j]
        state = None if prev == -1 else prev

    zs = np.arange(z0, z1)
    if len(chosen) < 8:
        chosen = {z0+i: cands[0] for i, cands in enumerate(raw) if cands}
    known_z = np.array(sorted(chosen), dtype=float)
    known_y = np.array([chosen[int(z)].y for z in known_z])
    known_x = np.array([chosen[int(z)].x for z in known_z])
    known_r = np.array([chosen[int(z)].radius_mm for z in known_z])
    if len(known_z) < 2:
        raise ValueError("Insufficient ascending-aorta detections")

    yi = ndi.gaussian_filter1d(np.interp(zs, known_z, known_y), sigma=2.0)
    xi = ndi.gaussian_filter1d(np.interp(zs, known_z, known_x), sigma=2.0)
    ri = ndi.gaussian_filter1d(np.interp(zs, known_z, known_r), sigma=2.0)

    track = {
        int(z): AortaDetection(int(z), float(y), float(x), float(r), 1.0, -1)
        for z, y, x, r in zip(zs, yi, xi, ri)
    }
    return track, (z0, z1)


def _slice_vessel_features(im_hu, spacing_xy_mm):
    im = np.asarray(im_hu, dtype=np.float32)
    sy, sx = float(spacing_xy_mm[1]), float(spacing_xy_mm[0])
    sm = ndi.gaussian_filter(im, sigma=0.75)
    norm = np.clip((sm + 150.0) / 950.0, 0.0, 1.4)
    vessel = frangi(norm, sigmas=(1.0, 1.5, 2.0, 2.7, 3.4), black_ridges=False)
    vessel = np.nan_to_num(vessel, nan=0.0, posinf=0.0, neginf=0.0)
    pos = vessel[vessel > 0]
    if pos.size:
        q = float(np.percentile(pos, 99.6))
        if q > 0:
            vessel = np.clip(vessel / q, 0.0, 1.0)
    bright = sm >= 160.0
    radius = ndi.distance_transform_edt(bright, sampling=(sy, sx))
    contrast = np.clip((sm - 150.0) / 300.0, 0.0, 1.0)
    return sm, vessel, radius, contrast


def find_rca_root_candidates(
    volume_hu,
    image,
    *,
    n=8,
    z_fraction=(0.46, 0.76),
    threshold_hu=260.0,
):
    """Find small persistent bright branches near the patient-right ascending aorta."""
    vol = np.asarray(volume_hu, dtype=np.float32)
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    sy, sx = float(spacing[1]), float(spacing[0])
    track, window = detect_ascending_aorta_track(
        vol, image, z_fraction=z_fraction, threshold_hu=threshold_hu
    )

    entries = []
    ygrid, xgrid = np.indices(vol.shape[1:])
    for z in range(window[0], window[1]):
        a = track[z]
        sm, vessel, radius, contrast = _slice_vessel_features(vol[z], spacing[:2])
        dmm = np.sqrt(((ygrid-a.y)*sy)**2 + ((xgrid-a.x)*sx)**2)
        shell = (dmm >= a.radius_mm + 0.3) & (dmm <= a.radius_mm + 11.0)
        small = (radius >= 0.45) & (radius <= 3.2)
        base = vessel * (0.25 + 0.75*contrast) * np.exp(-0.5*((radius-1.6)/1.0)**2)
        base[~(shell & small & (sm >= 170.0))] = 0.0

        mx = ndi.maximum_filter(base, size=7)
        peaks = np.argwhere((base == mx) & (base > 0.05))
        if len(peaks) == 0:
            continue
        vals = base[peaks[:,0], peaks[:,1]]
        order = np.argsort(vals)[::-1][:20]
        aphys = _physical_xyz(image, z, a.y, a.x)
        for idx in order:
            y, x = [int(v) for v in peaks[idx]]
            phys = _physical_xyz(image, z, y, x)
            if phys[0] > aphys[0] + 4.0:
                continue
            if phys[1] > aphys[1] + 14.0:
                continue

            support = 0
            for dz in range(-5, 6):
                zz = z + dz
                if zz < 0 or zz >= vol.shape[0]:
                    continue
                yy0, yy1 = max(0, y-4), min(vol.shape[1], y+5)
                xx0, xx1 = max(0, x-4), min(vol.shape[2], x+5)
                patch = vol[zz, yy0:yy1, xx0:xx1]
                if np.any(patch >= 180.0):
                    support += 1
            if support < 5:
                continue

            score = float(base[y, x] * (0.45 + 0.05*support))
            entries.append(RootCandidate(
                z=int(z), y=y, x=x, score=score,
                hu=float(vol[z,y,x]), vesselness=float(vessel[y,x]),
                local_radius_mm=float(radius[y,x]), support_slices=int(support),
                aorta_y=float(a.y), aorta_x=float(a.x), aorta_radius_mm=float(a.radius_mm),
            ))

    selected = []
    for c in sorted(entries, key=lambda q: q.score, reverse=True):
        ok = True
        for p in selected:
            dz_mm = abs(c.z-p.z) * float(spacing[2])
            dxy_mm = np.hypot((c.y-p.y)*sy, (c.x-p.x)*sx)
            if dz_mm < 2.0 and dxy_mm < 5.0:
                ok = False; break
        if ok:
            selected.append(c)
        if len(selected) >= int(n):
            break

    return selected, track, window
