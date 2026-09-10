from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .source_candidates import _slice_feature_map, find_root_window


@dataclass
class RCAOstiumCandidate:
    z: int
    y: int
    x: int
    score: float
    hu: float
    vesselness: float
    local_radius_mm: float
    aorta_y: float
    aorta_x: float
    aorta_radius_mm: float
    support_slices: int


def _pick_ascending_aorta(image_hu: np.ndarray):
    """Return a coarse ascending-aorta component for one axial CCTA slice.

    This is deliberately a geometric helper, not a diagnostic segmentation.
    It favors a large, round, contrast-filled component in the anterior-central
    part of the image and suppresses chest-wall / posterior structures.
    """
    im = ndi.gaussian_filter(np.asarray(image_hu, dtype=np.float32), sigma=1.0)
    h, w = im.shape
    mask = im >= 220.0

    # Keep only the central/anterior field where the ascending aorta is expected.
    roi = np.zeros_like(mask, dtype=bool)
    roi[int(0.14*h):int(0.70*h), int(0.16*w):int(0.72*w)] = True
    mask &= roi

    labels, n = ndi.label(mask)
    if n == 0:
        return None

    best = None
    best_score = -np.inf
    for lab in range(1, n + 1):
        yy, xx = np.where(labels == lab)
        area = len(yy)
        if area < 700 or area > 14000:
            continue
        cy, cx = float(yy.mean()), float(xx.mean())
        yspan = int(yy.max() - yy.min() + 1)
        xspan = int(xx.max() - xx.min() + 1)
        aspect = max(yspan, xspan) / max(1.0, min(yspan, xspan))
        if aspect > 2.0:
            continue

        # Ascending aorta is usually anterior-central in standard axial CCTA.
        dy = (cy - 0.42*h) / (0.22*h)
        dx = (cx - 0.40*w) / (0.22*w)
        center_score = np.exp(-0.5 * (dx*dx + dy*dy))
        area_score = np.exp(-0.5 * ((np.log(max(area, 1)) - np.log(3500.0)) / 0.75) ** 2)
        round_score = np.exp(-1.5 * (aspect - 1.0))
        score = 0.50*center_score + 0.30*area_score + 0.20*round_score
        if score > best_score:
            best_score = score
            best = (labels == lab, cy, cx, area)
    return best


def _candidate_on_slice(image_hu, spacing_yx_mm):
    picked = _pick_ascending_aorta(image_hu)
    if picked is None:
        return None
    aorta_mask, cy, cx, area = picked
    sy, sx = [float(v) for v in spacing_yx_mm]
    r_equiv_pix = float(np.sqrt(area / np.pi))
    r_equiv_mm = r_equiv_pix * float(np.sqrt(sx * sy))

    score_map, vesselness, radius_mm = _slice_feature_map(image_hu, (sy, sx))
    yy, xx = np.indices(image_hu.shape)
    dx_mm = (xx - cx) * sx
    dy_mm = (yy - cy) * sy
    radial_mm = np.sqrt(dx_mm*dx_mm + dy_mm*dy_mm)

    # Search just outside the aortic wall, on the patient's right/anterior side.
    # In standard radiologic axial display, patient-right is image-left (x < cx).
    annulus = (radial_mm >= max(0.75*r_equiv_mm, r_equiv_mm - 3.0)) & (radial_mm <= r_equiv_mm + 12.0)
    right_side = xx <= cx + 0.20*r_equiv_pix
    not_too_posterior = yy <= cy + 0.85*r_equiv_pix
    search = annulus & right_side & not_too_posterior

    # Remove the large aortic core; narrow coronary protrusions near the edge remain eligible.
    core = ndi.binary_erosion(aorta_mask, iterations=max(1, int(round(2.0 / max(sx, sy)))))
    search &= ~core

    local = np.array(score_map, copy=True)
    # Prefer points close to the wall but not inside the blood-pool center.
    target = r_equiv_mm + 2.5
    wall_weight = np.exp(-0.5 * ((radial_mm - target) / 5.0) ** 2)
    local *= wall_weight
    local[~search] = 0.0
    local[np.asarray(image_hu) < 120.0] = 0.0

    if not np.any(local > 0):
        return None
    iy, ix = np.unravel_index(int(np.argmax(local)), local.shape)
    return dict(
        y=int(iy), x=int(ix), base_score=float(local[iy, ix]),
        hu=float(image_hu[iy, ix]), vesselness=float(vesselness[iy, ix]),
        local_radius_mm=float(radius_mm[iy, ix]),
        aorta_y=float(cy), aorta_x=float(cx), aorta_radius_mm=float(r_equiv_mm),
    )


def find_rca_ostium_candidates(volume_hu, spacing_xyz_mm, *, n=6):
    """Find anatomy-constrained proximal-RCA/ostium candidates.

    The search is restricted to the estimated aortic-root window and to a narrow
    annulus around the ascending aorta on the patient-right/anterior side. Nearby
    slice-to-slice persistence boosts a candidate. Returned points are for visual
    review, not automatic clinical labeling.
    """
    vol = np.asarray(volume_hu, dtype=np.float32)
    spacing = np.asarray(spacing_xyz_mm, dtype=float)
    if vol.ndim != 3 or spacing.size != 3:
        raise ValueError("Expected a 3-D volume and xyz spacing")

    z0, z1 = find_root_window(vol, spacing)
    sy, sx = float(spacing[1]), float(spacing[0])
    raw = []
    for z in range(z0, z1):
        c = _candidate_on_slice(vol[z], (sy, sx))
        if c is not None:
            c["z"] = int(z)
            raw.append(c)

    if not raw:
        return [], (z0, z1)

    # Persistence: true proximal coronary candidates should recur over adjacent
    # 0.3-0.5 mm source slices at nearby in-plane positions.
    for c in raw:
        support = 0
        for d in raw:
            dz_mm = abs(d["z"] - c["z"]) * float(spacing[2])
            if dz_mm > 3.0:
                continue
            dxy = np.hypot((d["x"]-c["x"])*sx, (d["y"]-c["y"])*sy)
            if dxy <= 5.0:
                support += 1
        c["support"] = support
        c["score"] = c["base_score"] * (1.0 + 0.08 * min(support, 12))

    selected = []
    for c in sorted(raw, key=lambda q: q["score"], reverse=True):
        # Keep genuinely different hypotheses, not twelve adjacent copies.
        duplicate = False
        for p in selected:
            dz_mm = abs(c["z"] - p.z) * float(spacing[2])
            dxy = np.hypot((c["x"]-p.x)*sx, (c["y"]-p.y)*sy)
            if dz_mm < 4.0 and dxy < 6.0:
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(RCAOstiumCandidate(
            z=c["z"], y=c["y"], x=c["x"], score=float(c["score"]),
            hu=c["hu"], vesselness=c["vesselness"], local_radius_mm=c["local_radius_mm"],
            aorta_y=c["aorta_y"], aorta_x=c["aorta_x"], aorta_radius_mm=c["aorta_radius_mm"],
            support_slices=int(c["support"]),
        ))
        if len(selected) >= int(n):
            break

    return selected, (z0, z1)
