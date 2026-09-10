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

import torch
DEVICE = 'gpu' if torch.cuda.is_available() else 'cpu'
print('OpenPlaque branch: totalsegmentator-validation-from-main (based directly on main)')
print('Python:', sys.version.split()[0])
print('TotalSegmentator:', importlib.metadata.version('TotalSegmentator'))
print('nnunetv2:', importlib.metadata.version('nnunetv2'))
print('torch:', torch.__version__)
print('Device:', DEVICE)

# ---- Colab/Python 3.13 nnU-Net compatibility patch ----
import nnunetv2.inference.data_iterators as di
DI_PATH = Path(inspect.getfile(di))
text = DI_PATH.read_text()
patched = text.replace('manager = Manager()', 'manager = context.Manager()')
if patched != text:
    DI_PATH.write_text(patched)
    print('Patched nnU-Net Manager to use the same multiprocessing context.')
else:
    print('Manager-context patch already present or upstream changed.')

LICENSE_NUMBER = None
try:
    from google.colab import userdata
    try:
        LICENSE_NUMBER = userdata.get('TOTALSEG_LICENSE')
    except Exception:
        LICENSE_NUMBER = None
except Exception:
    pass
print('TOTALSEG_LICENSE secret found:', bool(LICENSE_NUMBER))


def enable_fork_preprocessing():
    txt = DI_PATH.read_text()
    new = txt.replace("multiprocessing.get_context('spawn')", "multiprocessing.get_context('fork')")
    new = new.replace('multiprocessing.get_context("spawn")', 'multiprocessing.get_context("fork")')
    if new != txt:
        DI_PATH.write_text(new)
        print('Fallback enabled: nnU-Net preprocessing now uses Linux fork.')
    else:
        print('No spawn preprocessing context remained to patch.')


def run_live(cmd, secret=None):
    safe = list(cmd)
    if secret and secret in safe:
        safe[safe.index(secret)] = '<hidden>'
    print('$', ' '.join(safe), flush=True)
    return subprocess.run(cmd).returncode


def run_totalseg(cmd, outdir, label):
    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f'\nRunning {label}, attempt 1 (normal spawn preprocessing)...', flush=True)
    t = time.time()
    rc = run_live(cmd, LICENSE_NUMBER)
    print(f'{label} attempt 1: {(time.time()-t)/60:.1f} min, return code={rc}')
    if rc == 0:
        return 0
    if sys.platform.startswith('linux'):
        print('\nAttempt 1 failed. Retrying once with Colab-safe preprocessing fallback.')
        enable_fork_preprocessing()
        shutil.rmtree(outdir, ignore_errors=True)
        outdir.mkdir(parents=True, exist_ok=True)
        t = time.time()
        rc = run_live(cmd, LICENSE_NUMBER)
        print(f'{label} attempt 2: {(time.time()-t)/60:.1f} min, return code={rc}')
    return rc

# ---- Load source series 7 from the stable main-branch study loader ----
ROOT = Path('/content/drive/MyDrive/OpenPlaque')
DRIVE_ZIP = ROOT / 'Full_DICOM.zip'
LOCAL_ZIP = Path('/content/Full_DICOM.zip')
EXTRACT_ROOT = '/content/full_dicom_totalseg_v2'
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
print('Series 7 shape z,y,x:', source.shape)
print('Series 7 spacing x,y,z:', source_img.GetSpacing())
INPUT_NII = Path('/content/source_series7.nii.gz')
sitk.WriteImage(source_img, str(INPUT_NII))

# ---- Capability check ----
subprocess.run(['totalseg_info', '--classes', '-ta', 'total'])
subprocess.run(['totalseg_info', '--classes', '-ta', 'coronary_arteries'])

# ---- Aorta ----
AORTA_DIR = Path('/content/totalseg_aorta_v2')
aorta_cmd = [
    'TotalSegmentator', '-i', str(INPUT_NII), '-o', str(AORTA_DIR),
    '-ta', 'total', '-rs', 'aorta', '--device', DEVICE,
    '-nr', '1', '-ns', '1'
]
rc_aorta = run_totalseg(aorta_cmd, AORTA_DIR, 'Aorta')
AORTA_AVAILABLE = (rc_aorta == 0)

# ---- Coronaries ----
COR_DIR = Path('/content/totalseg_coronary_v2')
CORONARY_AVAILABLE = False
if AORTA_AVAILABLE:
    cor_cmd = [
        'TotalSegmentator', '-i', str(INPUT_NII), '-o', str(COR_DIR),
        '-ta', 'coronary_arteries', '--device', DEVICE,
        '-nr', '1', '-ns', '1'
    ]
    if LICENSE_NUMBER:
        cor_cmd += ['-l', LICENSE_NUMBER]
    rc_cor = run_totalseg(cor_cmd, COR_DIR, 'Coronary arteries')
    CORONARY_AVAILABLE = (rc_cor == 0)
else:
    print('Aorta inference did not complete; skipping coronary inference.')

# ---- Load masks in source geometry ----
def find_mask(folder, preferred):
    p = folder / preferred
    if p.exists():
        return p
    files = sorted(folder.rglob('*.nii.gz')) if folder.exists() else []
    key = preferred.replace('.nii.gz', '').lower()
    hits = [q for q in files if key in q.name.lower()]
    return hits[0] if hits else (files[0] if files else None)


def read_mask(path):
    if path is None:
        return None
    img = sitk.ReadImage(str(path))
    same = (img.GetSize() == source_img.GetSize() and
            np.allclose(img.GetSpacing(), source_img.GetSpacing(), atol=1e-5) and
            np.allclose(img.GetOrigin(), source_img.GetOrigin(), atol=1e-4) and
            np.allclose(img.GetDirection(), source_img.GetDirection(), atol=1e-5))
    if not same:
        print('Resampling mask to source geometry:', path.name)
        img = sitk.Resample(img, source_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(img) > 0


aorta_path = find_mask(AORTA_DIR, 'aorta.nii.gz') if AORTA_AVAILABLE else None
aorta = read_mask(aorta_path)
cor_path = find_mask(COR_DIR, 'coronary_arteries.nii.gz') if CORONARY_AVAILABLE else None
cor = read_mask(cor_path)
if CORONARY_AVAILABLE and cor is None:
    CORONARY_AVAILABLE = False

print('\nAorta mask:', aorta_path)
print('Coronary mask:', cor_path)
if aorta is not None:
    print('Aorta voxels:', int(aorta.sum()))
if cor is not None:
    print('Coronary voxels:', int(cor.sum()))
    print('Coronary-containing slices:', int(np.sum(np.any(cor, axis=(1,2)))))

# ---- Static validation ----
OUTDIR = ROOT / 'TotalSegmentator_Validation_v2'
OUTDIR.mkdir(parents=True, exist_ok=True)
if aorta is None:
    print('\nNO AORTA MASK PRODUCED. Please send the final inference error text.')
    raise SystemExit(0)

counts = cor.sum(axis=(1,2)) if cor is not None else aorta.sum(axis=(1,2))
order = np.argsort(counts)[::-1]
zs = []
for z in order:
    if counts[z] <= 0:
        break
    if all(abs(int(z)-q) >= 7 for q in zs):
        zs.append(int(z))
    if len(zs) == 12:
        break
zs = sorted(zs)

fig, axes = plt.subplots(3,4,figsize=(16,12))
for ax in axes.ravel(): ax.axis('off')
for ax,z in zip(axes.ravel(), zs):
    ax.imshow(source[z], cmap='gray', vmin=-200, vmax=900)
    ax.contour(aorta[z].astype(float), levels=[0.5], linewidths=1.3)
    if cor is not None:
        ax.imshow(np.ma.masked_where(~cor[z], cor[z]), alpha=0.55)
    ax.set_title(f'z={z}')
    ax.axis('off')
plt.tight_layout()
p1 = OUTDIR / 'totalseg_v2_full_context.png'
fig.savefig(p1, dpi=180, bbox_inches='tight')
plt.show(); plt.close(fig)
print('Saved:', p1)

fig, axes = plt.subplots(3,4,figsize=(14,14))
for ax in axes.ravel(): ax.axis('off')
for ax,z in zip(axes.ravel(), zs):
    target = aorta[z] | (cor[z] if cor is not None else False)
    yy,xx = np.where(target)
    cy = int(np.median(yy)) if len(yy) else source.shape[1]//2
    cx = int(np.median(xx)) if len(xx) else source.shape[2]//2
    r=110; y0=max(0,cy-r); y1=min(source.shape[1],cy+r); x0=max(0,cx-r); x1=min(source.shape[2],cx+r)
    ax.imshow(source[z,y0:y1,x0:x1], cmap='gray', vmin=-200, vmax=900)
    ax.contour(aorta[z,y0:y1,x0:x1].astype(float), levels=[0.5], linewidths=1.3)
    if cor is not None:
        ax.imshow(np.ma.masked_where(~cor[z,y0:y1,x0:x1], cor[z,y0:y1,x0:x1]), alpha=0.60)
    ax.set_title(f'z={z}')
    ax.axis('off')
plt.tight_layout()
p2 = OUTDIR / 'totalseg_v2_closeups.png'
fig.savefig(p2, dpi=180, bbox_inches='tight')
plt.show(); plt.close(fig)
print('Saved:', p2)

spacing = np.array(source_img.GetSpacing(), float)
voxel_mm3 = float(np.prod(spacing))
rows = [{'mask':'aorta','voxels':int(aorta.sum()),'volume_ml':float(aorta.sum()*voxel_mm3/1000),'slices':int(np.sum(np.any(aorta,axis=(1,2))))}]
if cor is not None:
    rows.append({'mask':'coronary_arteries','voxels':int(cor.sum()),'volume_ml':float(cor.sum()*voxel_mm3/1000),'slices':int(np.sum(np.any(cor,axis=(1,2))))})
    iters=max(1,int(round(1.5/min(spacing[:2]))))
    near=cor & ndi.binary_dilation(aorta, iterations=iters)
    print('Coronary voxels within ~1.5 mm of aorta:', int(near.sum()))
    print('Slices with coronary-aorta proximity:', int(np.sum(np.any(near,axis=(1,2)))))
summary=pd.DataFrame(rows)
display(summary)
summary.to_csv(OUTDIR/'totalseg_v2_mask_summary.csv', index=False)
print('\nTOTAL SEGMENTATOR V2 VALIDATION COMPLETE.')
print('Please send totalseg_v2_full_context.png and totalseg_v2_closeups.png.')
