import os, sys, time, shutil, subprocess, inspect, importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy import ndimage as ndi

SRC = Path('/content/OpenPlaque/src')
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from openplaque.study import OpenPlaqueStudy

ROOT = Path('/content/drive/MyDrive/OpenPlaque')
OUTDIR = ROOT / 'RCA_Ostium_TotalSegmentator'
OUTDIR.mkdir(parents=True, exist_ok=True)
AORTA_CACHE = OUTDIR / 'aorta_series7_totalseg.nii.gz'

# ---------- Load the original best-diastolic CCTA (series 7) ----------
DRIVE_ZIP = ROOT / 'Full_DICOM.zip'
LOCAL_ZIP = Path('/content/Full_DICOM.zip')
EXTRACT_ROOT = '/content/full_dicom_totalseg_ostium'
if not DRIVE_ZIP.exists():
    raise FileNotFoundError(DRIVE_ZIP)
if not LOCAL_ZIP.exists() or LOCAL_ZIP.stat().st_size != DRIVE_ZIP.stat().st_size:
    print(f'Copying Full_DICOM.zip ({DRIVE_ZIP.stat().st_size/1e9:.2f} GB) to local disk...', flush=True)
    shutil.copyfile(DRIVE_ZIP, LOCAL_ZIP)
else:
    print('Local Full_DICOM.zip already staged.')

shutil.rmtree(EXTRACT_ROOT, ignore_errors=True)
print('Extracting/scanning DICOM locally...', flush=True)
study = OpenPlaqueStudy(str(LOCAL_ZIP), extract_root=EXTRACT_ROOT)
source_img, source, _ = study.load_series(7)
source = np.asarray(source)
spacing_xyz = np.array(source_img.GetSpacing(), float)
spacing_zyx = spacing_xyz[::-1]
voxel_mm3 = float(np.prod(spacing_xyz))
print('Series 7 shape z,y,x:', source.shape)
print('Series 7 spacing x,y,z:', tuple(spacing_xyz))

INPUT_NII = Path('/content/source_series7.nii.gz')
sitk.WriteImage(source_img, str(INPUT_NII))

# ---------- TotalSegmentator aorta mask, cached on Drive ----------
def geometry_matches(a, b):
    return (a.GetSize() == b.GetSize() and
            np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-5) and
            np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-4) and
            np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-5))


def read_mask_on_source(path):
    img = sitk.ReadImage(str(path))
    if not geometry_matches(img, source_img):
        print('Resampling aorta mask to source geometry...')
        img = sitk.Resample(img, source_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(img) > 0


def patch_nnunet_for_colab():
    # Colab Python 3.13 can lose the Manager connection when nnU-Net uses a spawned
    # preprocessing worker. Keep manager/process context consistent and use Linux fork.
    import nnunetv2.inference.data_iterators as di
    p = Path(inspect.getfile(di))
    txt = p.read_text()
    txt2 = txt.replace('manager = Manager()', 'manager = context.Manager()')
    txt2 = txt2.replace("multiprocessing.get_context('spawn')", "multiprocessing.get_context('fork')")
    txt2 = txt2.replace('multiprocessing.get_context("spawn")', 'multiprocessing.get_context("fork")')
    if txt2 != txt:
        p.write_text(txt2)
        print('Applied Colab-safe nnU-Net preprocessing patch.')
    else:
        print('nnU-Net preprocessing patch already present/upstream-compatible.')


if AORTA_CACHE.exists():
    print('Using cached TotalSegmentator aorta mask:', AORTA_CACHE)
    aorta = read_mask_on_source(AORTA_CACHE)
else:
    print('No cached aorta mask found; running TotalSegmentator once and caching it on Drive.')
    patch_nnunet_for_colab()
    import torch
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    work = Path('/content/totalseg_aorta_ostium')
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    cmd = [
        'TotalSegmentator', '-i', str(INPUT_NII), '-o', str(work),
        '-ta', 'total', '-rs', 'aorta', '--device', device,
        '-nr', '1', '-ns', '1'
    ]
    print('$', ' '.join(cmd), flush=True)
    t = time.time()
    rc = subprocess.run(cmd).returncode
    print(f'TotalSegmentator aorta runtime: {(time.time()-t)/60:.1f} min; rc={rc}')
    if rc != 0:
        raise RuntimeError('TotalSegmentator aorta segmentation failed.')
    aorta_path = work / 'aorta.nii.gz'
    if not aorta_path.exists():
        hits = sorted(work.rglob('*aorta*.nii.gz'))
        if not hits:
            raise FileNotFoundError('TotalSegmentator returned success but no aorta mask was found.')
        aorta_path = hits[0]
    aorta_img = sitk.ReadImage(str(aorta_path))
    if not geometry_matches(aorta_img, source_img):
        aorta_img = sitk.Resample(aorta_img, source_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    sitk.WriteImage(aorta_img, str(AORTA_CACHE))
    aorta = sitk.GetArrayFromImage(aorta_img) > 0
    print('Cached:', AORTA_CACHE)

print('Aorta voxels:', int(aorta.sum()), 'volume mL:', round(aorta.sum()*voxel_mm3/1000.0, 2))

# ---------- Physical-coordinate helpers ----------
origin = np.array(source_img.GetOrigin(), float)
direction = np.array(source_img.GetDirection(), float).reshape(3, 3)


def index_xyz_to_lps(x, y, z):
    idx_mm = np.array([x, y, z], float) * spacing_xyz
    return origin + direction @ idx_mm


def array_coord_to_lps(zyx):
    z, y, x = [float(v) for v in zyx]
    return index_xyz_to_lps(x, y, z)

# ---------- Identify the ascending-aorta run from the mask itself ----------
# At root levels the aorta mask normally has two large axial components:
# anterior ascending aorta and posterior descending aorta. We use physical LPS
# anterior/posterior coordinates, not display-left/right assumptions.
rows = []
components_by_z = {}
pix_area = float(spacing_xyz[0] * spacing_xyz[1])
for z in range(aorta.shape[0]):
    lab, n = ndi.label(aorta[z], structure=np.ones((3,3), bool))
    comps = []
    for k in range(1, n+1):
        yy, xx = np.where(lab == k)
        if len(xx) == 0:
            continue
        area = len(xx) * pix_area
        if area < 120.0 or area > 3000.0:
            continue
        cy, cx = float(np.mean(yy)), float(np.mean(xx))
        lps = index_xyz_to_lps(cx, cy, z)
        eq_r = np.sqrt(area / np.pi)
        comps.append({'label':k, 'area_mm2':area, 'r_mm':eq_r, 'cy':cy, 'cx':cx,
                      'lps_x':lps[0], 'lps_y':lps[1], 'lps_z':lps[2]})
    components_by_z[z] = comps
    if len(comps) >= 2:
        anterior = min(comps, key=lambda c: c['lps_y'])  # LPS y increases posteriorly
        posterior = max(comps, key=lambda c: c['lps_y'])
        sep = posterior['lps_y'] - anterior['lps_y']
        if 9.0 <= anterior['r_mm'] <= 28.0 and sep >= 12.0:
            rows.append({'z':z, **{f'a_{k}':v for k,v in anterior.items()}, 'ap_sep_mm':sep})

if not rows:
    raise RuntimeError('Could not identify slices with separate ascending and descending aorta components.')
track_df = pd.DataFrame(rows).sort_values('z').reset_index(drop=True)

# Split into runs allowing small segmentation gaps; choose the run with the greatest
# superior-inferior physical span and enough samples.
runs = []
cur = [int(track_df.iloc[0].z)]
for z in track_df.z.astype(int).tolist()[1:]:
    if z - cur[-1] <= 4:
        cur.append(z)
    else:
        runs.append(cur); cur = [z]
runs.append(cur)

def run_score(run):
    sub = track_df[track_df.z.isin(run)]
    span = float(sub.a_lps_z.max() - sub.a_lps_z.min())
    return abs(span) + 0.05*len(run)

run = max(runs, key=run_score)
run_df = track_df[track_df.z.isin(run)].copy().sort_values('a_lps_z')
if len(run_df) < 8:
    raise RuntimeError('Ascending-aorta component run was too short for reliable root localization.')

# Infer inferior -> superior from physical LPS z. Coronary ostia are near the inferior
# end of the robust ascending-aorta run. Ignore the first 1.5 mm of the mask tip, then
# search the next 24 mm.
s_min = float(run_df.a_lps_z.min())
root_search_lo = s_min + 1.5
root_search_hi = s_min + 25.5
root_df = run_df[(run_df.a_lps_z >= root_search_lo) & (run_df.a_lps_z <= root_search_hi)].copy()
if len(root_df) < 8:
    # Handle unusual sign/order by using the lower quarter of physical span.
    qlo, qhi = run_df.a_lps_z.quantile([0.02, 0.35])
    root_df = run_df[(run_df.a_lps_z >= qlo) & (run_df.a_lps_z <= qhi)].copy()

root_zs = sorted(root_df.z.astype(int).unique().tolist())
print(f'Ascending-aorta run: {len(run_df)} slices; root-search band: {len(root_zs)} slices')
print('Root-search z indices:', min(root_zs), 'to', max(root_zs))
print('Root-search physical S range mm:', round(root_df.a_lps_z.min(),1), 'to', round(root_df.a_lps_z.max(),1))

# Build an ascending-aorta-only mask in the root band.
asc = np.zeros_like(aorta, dtype=bool)
center_by_z = {}
for _, r in root_df.iterrows():
    z = int(r.z); k = int(r.a_label)
    lab, _ = ndi.label(aorta[z], structure=np.ones((3,3), bool))
    asc[z] = (lab == k)
    center_by_z[z] = np.array([r.a_lps_x, r.a_lps_y, r.a_lps_z], float)

# ---------- Root-band validation image ----------
show_zs = np.linspace(min(root_zs), max(root_zs), min(12, len(root_zs))).round().astype(int)
# snap to available root slices
show_zs = [min(root_zs, key=lambda q: abs(q-int(z))) for z in show_zs]
show_zs = list(dict.fromkeys(show_zs))
fig, axes = plt.subplots(3,4,figsize=(16,12))
for ax in axes.ravel(): ax.axis('off')
for ax,z in zip(axes.ravel(), show_zs):
    ax.imshow(source[z], cmap='gray', vmin=-200, vmax=900)
    ax.contour(aorta[z].astype(float), levels=[0.5], linewidths=1.0)
    ax.contour(asc[z].astype(float), levels=[0.5], linewidths=2.0)
    ax.set_title(f'root band z={z}')
    ax.axis('off')
plt.tight_layout()
p_root = OUTDIR / '01_aorta_root_band_validation.png'
fig.savefig(p_root, dpi=180, bbox_inches='tight')
plt.show(); plt.close(fig)
print('Saved:', p_root)

# ---------- Find contrast-filled tubular components emerging from root surface ----------
# Adaptive lumen threshold from the segmented ascending-aortic blood pool.
aorta_hu = source[asc]
if len(aorta_hu) == 0:
    raise RuntimeError('Ascending-aorta root mask is empty.')
med_hu = float(np.median(aorta_hu))
blood_thr = float(np.clip(0.43 * med_hu, 180.0, 360.0))
print('Median root-aorta HU:', round(med_hu,1), 'adaptive blood threshold:', round(blood_thr,1))

# Search a generous 3-D neighborhood, but remove a 0.9-mm guard around the aorta.
dist_to_asc = ndi.distance_transform_edt(~asc, sampling=spacing_zyx)
root_neighborhood = dist_to_asc <= 22.0
guard = dist_to_asc <= 0.9

# Restrict superior/inferior range to the root search band plus 5 mm margin in physical S.
s_lo = float(root_df.a_lps_z.min()) - 5.0
s_hi = float(root_df.a_lps_z.max()) + 5.0
slice_s = np.array([index_xyz_to_lps(0,0,z)[2] for z in range(source.shape[0])])
s_band = (slice_s >= min(s_lo,s_hi)) & (slice_s <= max(s_lo,s_hi))
root_neighborhood &= s_band[:,None,None]

blood = (source >= blood_thr) & root_neighborhood & (~guard)
# One gentle closing reconnects partially volumed coronary lumen without merging distant vessels.
blood = ndi.binary_closing(blood, structure=np.ones((3,3,3), bool), iterations=1)
lab3, n3 = ndi.label(blood, structure=np.ones((3,3,3), bool))
print('3-D high-contrast components in root neighborhood:', n3)

candidate_rows = []
component_masks = {}
for k in range(1, n3+1):
    coords = np.argwhere(lab3 == k)
    nv = len(coords)
    if nv < 5:
        continue
    vol = nv * voxel_mm3
    if vol > 5000.0:
        continue

    dvals = dist_to_asc[tuple(coords.T)]
    min_d = float(np.min(dvals)); max_d = float(np.max(dvals))
    if min_d > 3.0 or max_d < 1.8:
        continue

    # Physical-shape PCA (translation irrelevant).
    mm = coords.astype(float) * spacing_zyx[None,:]
    if len(mm) >= 3:
        ev = np.linalg.eigvalsh(np.cov(mm, rowvar=False))
        ev = np.sort(np.maximum(ev, 1e-6))[::-1]
    else:
        ev = np.array([1e-6,1e-6,1e-6])
    length_est = float(np.sqrt(12.0 * ev[0]))
    elong = float(ev[0] / max(ev[1], 1e-6))
    n_slices = int(np.unique(coords[:,0]).size)

    # Local radius distinguishes a coronary-sized tube from pulmonary artery/chambers.
    z0,y0,x0 = coords.min(axis=0); z1,y1,x1 = coords.max(axis=0)+1
    sub = (lab3[z0:z1,y0:y1,x0:x1] == k)
    local_r = ndi.distance_transform_edt(sub, sampling=spacing_zyx)
    max_radius = float(local_r.max())

    # Contact estimate = component voxels closest to the segmented aortic surface.
    near_cut = min_d + max(0.8, float(min(spacing_xyz))*2.0)
    near_coords = coords[dvals <= near_cut]
    contact = np.median(near_coords, axis=0)
    contact_lps = array_coord_to_lps(contact)
    zc = int(round(contact[0]))
    z_ref = min(root_zs, key=lambda q: abs(q-zc))
    a_center = center_by_z[z_ref]
    dx = float(contact_lps[0] - a_center[0])  # negative = patient-right in LPS
    dy = float(contact_lps[1] - a_center[1])  # negative = anterior in LPS
    side = 'RCA-like' if dx < 0 else 'LM-like'

    huvals = source[tuple(coords.T)]
    mean_hu = float(np.mean(huvals))
    # Continuous score emphasizes coronary caliber + tubular continuation away from aortic wall.
    tube = np.tanh(length_est/10.0)
    ext = np.tanh(max_d/8.0)
    elong_s = np.tanh(elong/5.0)
    slice_supp = np.tanh(n_slices/8.0)
    proximity = np.exp(-min_d/1.5)
    radius_pref = np.exp(-((max_radius-2.2)/2.2)**2)
    base = 0.22*tube + 0.20*ext + 0.18*elong_s + 0.14*slice_supp + 0.12*proximity + 0.14*radius_pref
    right_prior = np.tanh(max(0.0, -dx)/8.0)
    anterior_prior = np.tanh(max(0.0, -dy)/8.0)
    rca_score = float(base + 0.12*right_prior + 0.04*anterior_prior)
    lm_score = float(base + 0.12*np.tanh(max(0.0, dx)/8.0) + 0.02*anterior_prior)

    strict = (2.0 <= length_est and n_slices >= 2 and max_radius <= 5.5 and
              max_d >= 2.5 and vol <= 1800.0)
    candidate_rows.append({
        'label':k, 'side':side, 'base_score':base, 'rca_score':rca_score, 'lm_score':lm_score,
        'z_contact':float(contact[0]), 'y_contact':float(contact[1]), 'x_contact':float(contact[2]),
        'lps_x':contact_lps[0], 'lps_y':contact_lps[1], 'lps_z':contact_lps[2],
        'dx_from_aorta_mm':dx, 'dy_from_aorta_mm':dy,
        'volume_mm3':vol, 'length_est_mm':length_est, 'elongation':elong,
        'slices':n_slices, 'min_dist_aorta_mm':min_d, 'max_dist_aorta_mm':max_d,
        'max_radius_mm':max_radius, 'mean_hu':mean_hu, 'strict':strict
    })
    component_masks[k] = (z0,z1,y0,y1,x0,x1,sub)

cand = pd.DataFrame(candidate_rows)
if cand.empty:
    raise RuntimeError('No near-aortic high-contrast components were found. Inspect the root-band image.')

strict = cand[cand.strict].copy()
if len(strict) < 2:
    print('Strict coronary-size filter yielded fewer than two candidates; using relaxed ranking for display.')
    strict = cand[(cand.max_radius_mm <= 7.0) & (cand.volume_mm3 <= 3000.0) &
                  (cand.length_est_mm >= 1.5)].copy()
if strict.empty:
    strict = cand.copy()

# Keep spatially distinct contact hypotheses.
def distinct_top(df, score_col, n=5, min_sep_mm=4.0):
    chosen = []
    for _, r in df.sort_values(score_col, ascending=False).iterrows():
        p = np.array([r.lps_x,r.lps_y,r.lps_z], float)
        if all(np.linalg.norm(p-q[0]) >= min_sep_mm for q in chosen):
            chosen.append((p,r))
        if len(chosen) >= n:
            break
    return [r for _,r in chosen]

rca_pool = strict[strict.dx_from_aorta_mm < 1.5]
lm_pool = strict[strict.dx_from_aorta_mm > -1.5]
rca_top = distinct_top(rca_pool if len(rca_pool) else strict, 'rca_score', n=5)
lm_top = distinct_top(lm_pool if len(lm_pool) else strict, 'lm_score', n=3)
selected = []
seen = set()
for tag, seq in [('R',rca_top),('L',lm_top)]:
    j=1
    for r in seq:
        k=int(r.label)
        if k in seen: continue
        selected.append((f'{tag}{j}',r)); seen.add(k); j+=1

cand.sort_values('rca_score', ascending=False).to_csv(OUTDIR/'ostium_candidate_metrics.csv', index=False)
print('\nTop candidate metrics:')
cols=['label','side','rca_score','lm_score','length_est_mm','max_radius_mm','max_dist_aorta_mm','dx_from_aorta_mm','dy_from_aorta_mm','mean_hu','strict']
display(cand.sort_values('rca_score',ascending=False)[cols].head(12))

# ---------- Candidate gallery: full context + close-up ----------
def component_slice_mask(k,z):
    return (lab3[z] == int(k))

fig, axes = plt.subplots(len(selected), 2, figsize=(12, 4.2*len(selected)))
if len(selected) == 1:
    axes = np.array([axes])
for row_i,(name,r) in enumerate(selected):
    z = int(round(r.z_contact)); y=int(round(r.y_contact)); x=int(round(r.x_contact)); k=int(r.label)
    for col in range(2):
        ax=axes[row_i,col]
        if col==0:
            ax.imshow(source[z], cmap='gray', vmin=-200, vmax=900)
            ax.contour(aorta[z].astype(float), levels=[0.5], linewidths=1.2)
            cm=component_slice_mask(k,z)
            if cm.any(): ax.contour(cm.astype(float), levels=[0.5], linewidths=2.0)
            ax.plot(x,y,'o',ms=6)
            ax.set_title(f'{name} full context z={z}')
        else:
            rad=70; y0=max(0,y-rad);y1=min(source.shape[1],y+rad);x0=max(0,x-rad);x1=min(source.shape[2],x+rad)
            ax.imshow(source[z,y0:y1,x0:x1], cmap='gray', vmin=-200, vmax=900)
            try: ax.contour(aorta[z,y0:y1,x0:x1].astype(float), levels=[0.5], linewidths=1.2)
            except Exception: pass
            cm=component_slice_mask(k,z)[y0:y1,x0:x1]
            if cm.any(): ax.contour(cm.astype(float), levels=[0.5], linewidths=2.0)
            ax.plot(x-x0,y-y0,'o',ms=6)
            ax.set_title(f'{name}: score={r.rca_score if name.startswith("R") else r.lm_score:.3f}  '
                         f'L~{r.length_est_mm:.1f}mm r={r.max_radius_mm:.1f}mm  dX={r.dx_from_aorta_mm:.1f}')
        ax.axis('off')
plt.tight_layout()
p_candidates = OUTDIR/'02_ostium_candidate_gallery.png'
fig.savefig(p_candidates,dpi=180,bbox_inches='tight')
plt.show(); plt.close(fig)
print('Saved:', p_candidates)

# ---------- Orthogonal/thin-slab views of the best two hypotheses ----------
best_views = []
if rca_top: best_views.append(('Best RCA-like', rca_top[0]))
if lm_top:
    # avoid duplicate if pools overlap
    if not best_views or int(lm_top[0].label) != int(best_views[0][1].label):
        best_views.append(('Best LM-like', lm_top[0]))

fig, axes = plt.subplots(len(best_views), 3, figsize=(15,5*len(best_views)))
if len(best_views)==1: axes=np.array([axes])
for i,(title,r) in enumerate(best_views):
    z=int(round(r.z_contact)); y=int(round(r.y_contact)); x=int(round(r.x_contact)); k=int(r.label)
    # axial thin-slab MIP +/- 3 mm
    dz=max(1,int(round(3.0/spacing_xyz[2]))); z0=max(0,z-dz);z1=min(source.shape[0],z+dz+1)
    mip=np.max(source[z0:z1],axis=0)
    cm=np.any(lab3[z0:z1]==k,axis=0)
    ax=axes[i,0]; ax.imshow(mip,cmap='gray',vmin=-200,vmax=900); ax.contour(cm.astype(float),levels=[0.5],linewidths=2.0); ax.plot(x,y,'o',ms=6); ax.set_title(title+' axial MIP ±3 mm'); ax.axis('off')
    # coronal and sagittal local MIPs around contact
    rz=max(1,int(round(16/spacing_xyz[2]))); ry=max(1,int(round(18/spacing_xyz[1]))); rx=max(1,int(round(18/spacing_xyz[0])))
    zz0=max(0,z-rz);zz1=min(source.shape[0],z+rz+1); yy0=max(0,y-ry);yy1=min(source.shape[1],y+ry+1); xx0=max(0,x-rx);xx1=min(source.shape[2],x+rx+1)
    crop=source[zz0:zz1,yy0:yy1,xx0:xx1]
    ccomp=(lab3[zz0:zz1,yy0:yy1,xx0:xx1]==k)
    cor=np.max(crop,axis=1); cor_m=np.any(ccomp,axis=1)
    sag=np.max(crop,axis=2); sag_m=np.any(ccomp,axis=2)
    ax=axes[i,1]; ax.imshow(cor,cmap='gray',vmin=-200,vmax=900,aspect='auto');
    if cor_m.any(): ax.contour(cor_m.astype(float),levels=[0.5],linewidths=2.0)
    ax.set_title(title+' local coronal MIP'); ax.axis('off')
    ax=axes[i,2]; ax.imshow(sag,cmap='gray',vmin=-200,vmax=900,aspect='auto');
    if sag_m.any(): ax.contour(sag_m.astype(float),levels=[0.5],linewidths=2.0)
    ax.set_title(title+' local sagittal MIP'); ax.axis('off')
plt.tight_layout()
p_views=OUTDIR/'03_best_ostium_orthogonal_views.png'
fig.savefig(p_views,dpi=180,bbox_inches='tight')
plt.show(); plt.close(fig)
print('Saved:', p_views)

print('\nOSTIUM-CANDIDATE VALIDATION COMPLETE.')
print('Please send these three images:')
print('  01_aorta_root_band_validation.png')
print('  02_ostium_candidate_gallery.png')
print('  03_best_ostium_orthogonal_views.png')
print('No centerline has been accepted or generated yet; the next step depends on visual validation of these candidates.')
