from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import frangi


@dataclass
class AortaSlice:
    z: int
    y: float
    x: float
    radius_mm: float
    area_mm2: float
    mean_hu: float
    lps_x_mm: float
    lps_y_mm: float
    mask: np.ndarray
    confidence: float


@dataclass
class RCAWallCandidate:
    z: int
    y: int
    x: int
    score: float
    hu: float
    vesselness: float
    radius_mm: float
    distance_from_aorta_mm: float
    support_slices: int
    aorta_y: float
    aorta_x: float
    aorta_radius_mm: float


def _physical_xy(image, z: int, y: float, x: float) -> tuple[float, float]:
    p = image.TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z)))
    return float(p[0]), float(p[1])


def _blood_pool_candidates(image_hu: np.ndarray, z: int, image, spacing_xy_mm) -> list[dict]:
    """Large, round, arterial-phase blood-pool candidates on one axial slice."""
    im = np.asarray(image_hu, dtype=np.float32)
    sy, sx = float(spacing_xy_mm[0]), float(spacing_xy_mm[1])
    h, w = im.shape
    sm = ndi.gaussian_filter(im, sigma=1.0, mode="nearest")
    mask = sm >= 260.0
    mask = ndi.binary_closing(mask, iterations=1)

    # Keep the anterior-central mediastinum. This excludes the descending aorta,
    # most chamber volume, chest wall, and lateral pulmonary branches.
    roi = np.zeros_like(mask, dtype=bool)
    roi[int(0.16*h):int(0.64*h), int(0.18*w):int(0.74*w)] = True
    mask &= roi

    labels, n = ndi.label(mask)
    out = []
    for lab in range(1, n + 1):
        yy, xx = np.where(labels == lab)
        if len(yy) == 0:
            continue
        area_mm2 = float(len(yy) * sy * sx)
        if not (220.0 <= area_mm2 <= 1600.0):
            continue
        y0, y1 = int(yy.min()), int(yy.max())
        x0, x1 = int(xx.min()), int(xx.max())
        hbox, wbox = y1-y0+1, x1-x0+1
        aspect = max(hbox*sy, wbox*sx) / max(1e-6, min(hbox*sy, wbox*sx))
        if aspect > 1.85:
            continue
        fill = float(len(yy) / max(1, hbox*wbox))
        if fill < 0.45:
            continue
        cy, cx = float(yy.mean()), float(xx.mean())
        r = float(np.sqrt(area_mm2 / np.pi))
        lps_x, lps_y = _physical_xy(image, z, cy, cx)
        mean_hu = float(np.mean(im[yy, xx]))

        # Prefer an aorta-sized, compact, high-contrast component.
        size_score = float(np.exp(-0.5 * ((r - 16.0) / 5.0) ** 2))
        round_score = float(np.exp(-1.5 * (aspect - 1.0)))
        fill_score = float(np.clip((fill - 0.45) / 0.35, 0.0, 1.0))
        hu_score = float(np.clip((mean_hu - 250.0) / 350.0, 0.0, 1.0))
        shape_score = 0.40*size_score + 0.30*round_score + 0.15*fill_score + 0.15*hu_score
        out.append(dict(
            z=int(z), y=cy, x=cx, radius_mm=r, area_mm2=area_mm2,
            mean_hu=mean_hu, lps_x=lps_x, lps_y=lps_y,
            mask=(labels == lab), shape_score=float(shape_score),
        ))
    return out


def _bootstrap_aorta(candidates_by_z: dict[int, list[dict]], spacing_z_mm: float):
    """Use slices containing an aorta/pulmonary pair to bootstrap the aortic track.

    In LPS coordinates, patient-right has a smaller x coordinate. Among two
    central, similarly sized great-vessel candidates, the patient-right member is
    therefore the ascending aorta.
    """
    seeds = []
    for z, cs in candidates_by_z.items():
        if len(cs) < 2:
            continue
        pairs = []
        for i in range(len(cs)):
            for j in range(i+1, len(cs)):
                a, b = cs[i], cs[j]
                d = float(np.hypot(a['lps_x']-b['lps_x'], a['lps_y']-b['lps_y']))
                if not (14.0 <= d <= 55.0):
                    continue
                ratio = max(a['radius_mm'], b['radius_mm']) / max(1e-6, min(a['radius_mm'], b['radius_mm']))
                if ratio > 1.8:
                    continue
                right = a if a['lps_x'] < b['lps_x'] else b
                pair_score = right['shape_score'] - 0.015*abs(d-28.0)
                pairs.append((pair_score, right))
        if pairs:
            seeds.append(max(pairs, key=lambda q: q[0])[1])

    if not seeds:
        return None

    # Find the densest spatial cluster of pair-derived patient-right candidates.
    xy = np.array([[s['lps_x'], s['lps_y']] for s in seeds], dtype=float)
    support = []
    for i, q in enumerate(xy):
        d = np.linalg.norm(xy-q, axis=1)
        support.append(int(np.sum(d <= 14.0)))
    best_i = int(np.argmax(support))
    keep = np.linalg.norm(xy-xy[best_i], axis=1) <= 14.0
    cluster = [s for s, k in zip(seeds, keep) if k]
    if len(cluster) < 3:
        cluster = seeds
    return cluster


def track_ascending_aorta(volume_hu, image, *, z_fraction=(0.40, 0.82)) -> list[AortaSlice]:
    """Track the ascending aorta without using 3-D blood-pool connectivity."""
    vol = np.asarray(volume_hu, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError("volume_hu must be 3-D")
    spacing_xyz = np.asarray(image.GetSpacing(), dtype=float)
    sx, sy, sz = [float(v) for v in spacing_xyz]
    z0 = max(0, int(round(vol.shape[0]*float(z_fraction[0]))))
    z1 = min(vol.shape[0], int(round(vol.shape[0]*float(z_fraction[1]))))

    candidates_by_z = {}
    for z in range(z0, z1):
        cs = _blood_pool_candidates(vol[z], z, image, (sy, sx))
        if cs:
            candidates_by_z[z] = cs

    cluster = _bootstrap_aorta(candidates_by_z, sz)
    if not cluster:
        raise ValueError("Could not bootstrap ascending aorta from great-vessel pairs")

    seed_x = float(np.median([s['lps_x'] for s in cluster]))
    seed_y = float(np.median([s['lps_y'] for s in cluster]))
    seed_r = float(np.median([s['radius_mm'] for s in cluster]))

    # Select a spatially coherent aortic candidate independently on each slice,
    # then retain the longest physically continuous run.
    raw = []
    for z in range(z0, z1):
        cs = candidates_by_z.get(z, [])
        if not cs:
            raw.append(None)
            continue
        ranked = []
        for c in cs:
            d = float(np.hypot(c['lps_x']-seed_x, c['lps_y']-seed_y))
            dr = abs(c['radius_mm']-seed_r)
            # Strongly prefer the patient-right cluster established by the pair bootstrap.
            score = c['shape_score'] - 0.055*d - 0.035*dr
            ranked.append((score, c, d))
        score, c, d = max(ranked, key=lambda q: q[0])
        if d > 22.0:
            raw.append(None)
            continue
        raw.append((score, c))

    # Longest run allowing short gaps; gaps are not emitted as anchors.
    best_run, cur = [], []
    gap = 0
    for idx, item in enumerate(raw):
        if item is None:
            gap += 1
            if gap <= max(2, int(round(1.5/sz))):
                cur.append((idx, None))
            else:
                valid = [q for q in cur if q[1] is not None]
                if len(valid) > len(best_run):
                    best_run = valid
                cur = []
                gap = 0
        else:
            gap = 0
            cur.append((idx, item))
    valid = [q for q in cur if q[1] is not None]
    if len(valid) > len(best_run):
        best_run = valid
    if len(best_run) < 8:
        raise ValueError("Ascending-aorta track was too short or unstable")

    out = []
    for idx, (score, c) in best_run:
        conf = float(np.clip(0.5 + 0.5*score, 0.0, 1.0))
        out.append(AortaSlice(
            z=int(c['z']), y=float(c['y']), x=float(c['x']), radius_mm=float(c['radius_mm']),
            area_mm2=float(c['area_mm2']), mean_hu=float(c['mean_hu']),
            lps_x_mm=float(c['lps_x']), lps_y_mm=float(c['lps_y']), mask=c['mask'], confidence=conf,
        ))
    return out


def _vessel_feature_map(im):
    arr = np.asarray(im, dtype=np.float32)
    sm = ndi.gaussian_filter(arr, sigma=0.7, mode='nearest')
    norm = np.clip((sm + 100.0)/900.0, 0.0, 1.5)
    ves = frangi(norm, sigmas=(1.0,1.5,2.0,2.8), black_ridges=False)
    ves = np.nan_to_num(ves, nan=0.0, posinf=0.0, neginf=0.0)
    p = ves[ves>0]
    if p.size:
        s = np.percentile(p, 99.5)
        if s > 0:
            ves = np.clip(ves/s, 0.0, 1.0)
    return sm, ves


def find_rca_wall_candidates(volume_hu, image, aorta_track: list[AortaSlice], *, n=8) -> list[RCAWallCandidate]:
    """Find persistent small bright vessels leaving the right-anterior aortic wall."""
    vol = np.asarray(volume_hu, dtype=np.float32)
    sx, sy, sz = [float(v) for v in image.GetSpacing()]
    by_z = {a.z: a for a in aorta_track}
    if len(by_z) < 8:
        return []

    # Coronary ostia lie toward the inferior/root end of the ascending aortic track.
    track_phys_z = [(a.z, image.TransformContinuousIndexToPhysicalPoint((a.x,a.y,float(a.z)))[2]) for a in aorta_track]
    phys_vals = np.asarray([p for _,p in track_phys_z], dtype=float)
    lo_phys, hi_phys = np.percentile(phys_vals, [0, 58])

    raw = []
    for a in aorta_track:
        pz = float(image.TransformContinuousIndexToPhysicalPoint((a.x,a.y,float(a.z)))[2])
        if not (lo_phys-1e-6 <= pz <= hi_phys+1e-6):
            continue
        sm, ves = _vessel_feature_map(vol[a.z])
        blood = sm >= 150.0
        rad = ndi.distance_transform_edt(blood, sampling=(sy, sx))
        outside_dist = ndi.distance_transform_edt(~a.mask, sampling=(sy, sx))

        yy, xx = np.indices(sm.shape)
        # Convert in-plane displacement to LPS using image geometry, so right/anterior
        # means lower LPS x and lower LPS y irrespective of display orientation.
        center_pt = np.asarray(image.TransformContinuousIndexToPhysicalPoint((a.x,a.y,float(a.z))))
        # For axial source data these direction vectors are constant across the slice.
        px = np.asarray(image.TransformContinuousIndexToPhysicalPoint((a.x+1.0,a.y,float(a.z)))) - center_pt
        py = np.asarray(image.TransformContinuousIndexToPhysicalPoint((a.x,a.y+1.0,float(a.z)))) - center_pt
        dx_lps = (xx-a.x)*px[0] + (yy-a.y)*py[0]
        dy_lps = (xx-a.x)*px[1] + (yy-a.y)*py[1]

        shell = (outside_dist >= 0.3) & (outside_dist <= 8.0)
        right = dx_lps <= 3.0
        anterior = dy_lps <= 5.0
        small = (rad >= 0.45) & (rad <= 3.4)
        bright = sm >= 170.0
        search = shell & right & anterior & small & bright & (~a.mask)
        if not np.any(search):
            continue

        wall_weight = np.exp(-0.5*((outside_dist-2.5)/2.5)**2)
        caliber_weight = np.exp(-0.5*((rad-1.7)/1.2)**2)
        hu_weight = np.clip((sm-150.0)/400.0, 0.0, 1.0)
        score_map = ves * (0.35+0.65*hu_weight) * caliber_weight * wall_weight
        score_map[~search] = 0.0
        if not np.any(score_map > 0):
            continue
        iy, ix = np.unravel_index(int(np.argmax(score_map)), score_map.shape)
        raw.append(dict(
            z=a.z, y=int(iy), x=int(ix), base=float(score_map[iy,ix]),
            hu=float(sm[iy,ix]), vesselness=float(ves[iy,ix]), radius=float(rad[iy,ix]),
            da=float(outside_dist[iy,ix]), a=a,
        ))

    # Slice persistence around the same physical point.
    for c in raw:
        p = np.asarray(image.TransformContinuousIndexToPhysicalPoint((float(c['x']),float(c['y']),float(c['z']))))
        support = 0
        for d in raw:
            if abs(d['z']-c['z'])*sz > 3.0:
                continue
            q = np.asarray(image.TransformContinuousIndexToPhysicalPoint((float(d['x']),float(d['y']),float(d['z']))))
            if np.linalg.norm(q-p) <= 5.0:
                support += 1
        c['support'] = support
        c['score'] = c['base'] * (1.0 + 0.10*min(support,12))

    selected = []
    for c in sorted(raw, key=lambda q:q['score'], reverse=True):
        p = np.asarray(image.TransformContinuousIndexToPhysicalPoint((float(c['x']),float(c['y']),float(c['z']))))
        duplicate = False
        for s in selected:
            q = np.asarray(image.TransformContinuousIndexToPhysicalPoint((float(s.x),float(s.y),float(s.z))))
            if np.linalg.norm(q-p) < 6.0:
                duplicate = True
                break
        if duplicate:
            continue
        a = c['a']
        selected.append(RCAWallCandidate(
            z=int(c['z']), y=int(c['y']), x=int(c['x']), score=float(c['score']),
            hu=float(c['hu']), vesselness=float(c['vesselness']), radius_mm=float(c['radius']),
            distance_from_aorta_mm=float(c['da']), support_slices=int(c['support']),
            aorta_y=float(a.y), aorta_x=float(a.x), aorta_radius_mm=float(a.radius_mm),
        ))
        if len(selected) >= int(n):
            break
    return selected
