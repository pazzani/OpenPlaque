import os, sys, shutil, zipfile, base64, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path('/content/OpenPlaque')
sys.path.insert(0, str(REPO / 'src'))
ROOT = Path('/content/drive/MyDrive/OpenPlaque')
OUT = ROOT / 'Image_Driven_Coronary_Tracking_Report'
OUT.mkdir(parents=True, exist_ok=True)

os.environ['nnUNet_raw'] = '/content/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '/content/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '/content/nnUNet_results'
for d in (os.environ['nnUNet_raw'], os.environ['nnUNet_preprocessed'], os.environ['nnUNet_results']):
    Path(d).mkdir(parents=True, exist_ok=True)

model_zip = ROOT / 'models' / 'Dataset001_CCTA_DHM-20260703T233210Z-3-001.zip'
model_target = Path('/content/nnUNet_results/Dataset001_CCTA_DHM')
if not model_target.exists():
    if not model_zip.exists():
        raise FileNotFoundError(model_zip)
    with zipfile.ZipFile(model_zip) as z:
        z.extractall('/content/nnUNet_results')

drive_zip = ROOT / 'Full_DICOM.zip'
local_zip = Path('/content/Full_DICOM.zip')
if not drive_zip.exists():
    raise FileNotFoundError(drive_zip)
if not local_zip.exists() or local_zip.stat().st_size != drive_zip.stat().st_size:
    shutil.copyfile(drive_zip, local_zip)

from openplaque.study import OpenPlaqueStudy
from openplaque.segmentation import segment_vessel
from openplaque.boundary import refine_plaque_mask
from openplaque.artery_detection import detect_artery_series
from openplaque.cpr_tracking import track_frame, path_tube, straighten

shutil.rmtree('/content/full_dicom_tracking', ignore_errors=True)
study = OpenPlaqueStudy(str(local_zip), extract_root='/content/full_dicom_tracking')

fallback = {'RCA': 1035, 'LCX': 1039, 'LAD': 1043}
series_map, _ = detect_artery_series(study, fallback_series=fallback, return_candidates=True)

reports = []
for vessel in ['LAD', 'RCA', 'LCX']:
    image, volume, _ = study.load_series(series_map[vessel])
    print('Segmenting', vessel, 'series', series_map[vessel])
    reports.append(segment_vessel(image, volume, vessel))


def canonical_refine(r):
    return refine_plaque_mask(
        volume=r.volume,
        mask=r.mask,
        spacing=r.mask_image.GetSpacing(),
        remove_small=True,
        min_component_voxels=10,
        trim_lumen_adjacent=True,
        lumen_distance_voxels=1,
        erode_core=False,
        high_hu_threshold=None,
        low_hu_threshold=None,
    )


canonical = {r.name: canonical_refine(r) for r in reports}
all_candidates = {}
selected = {}
qc_rows = []

for r in reports:
    spacing = r.mask_image.GetSpacing()
    sp_yx = (float(spacing[1]), float(spacing[0]))
    plaque3 = canonical[r.name].refined_mask == 2
    rows = []

    for z in range(r.volume.shape[0]):
        img = np.asarray(r.volume[z], float)
        tr = track_frame(img, sp_yx)
        if tr is None:
            continue
        tube, path_mask = path_tube(tr['path'], img.shape, sp_yx, radius_mm=5.0)
        tr.update(
            frame=int(z),
            tube=tube,
            path_mask=path_mask,
            plaque_tube_voxels=int((plaque3[z] & tube).sum()),
            plaque_frame_voxels=int(plaque3[z].sum()),
        )
        rows.append(tr)

    if not rows:
        raise RuntimeError(f'No image-driven path found for {r.name}')

    rows = sorted(rows, key=lambda d: d['image_score'], reverse=True)
    top = rows[:min(5, len(rows))]
    max_image = max(x['image_score'] for x in top)
    max_plaque = max(x['plaque_tube_voxels'] for x in top)
    for x in top:
        image_norm = x['image_score'] / max(max_image, 1e-9)
        plaque_norm = x['plaque_tube_voxels'] / max(max_plaque, 1) if max_plaque > 0 else 0.0
        x['selection_score'] = image_norm + 0.10 * plaque_norm

    win = max(top, key=lambda d: d['selection_score'])
    all_candidates[r.name] = rows
    selected[r.name] = win
    qc_rows.append({
        'vessel': r.name,
        'selected_frame': win['frame'],
        'image_score': win['image_score'],
        'selection_score': win['selection_score'],
        'path_length_mm': win['path_length_mm'],
        'mean_path_hu': win['mean_path_hu'],
        'mean_evidence': win['mean_evidence'],
        'threshold_percentile': win['threshold_percentile'],
        'plaque_voxels_in_selected_frame': win['plaque_frame_voxels'],
        'plaque_voxels_within_5mm_tube': win['plaque_tube_voxels'],
        'tube_capture_pct_of_selected_frame': 100 * win['plaque_tube_voxels'] / max(1, win['plaque_frame_voxels']),
        'tracking_status': 'OK' if win['path_length_mm'] >= 25 and 100 <= win['mean_path_hu'] <= 800 else 'REVIEW',
    })

tracking_qc = pd.DataFrame(qc_rows)
tracking_qc.to_csv(OUT / 'tracking_qc.csv', index=False)
print(tracking_qc.to_string(index=False))

# Diagnostic: top three image-driven candidates per vessel.
fig, axs = plt.subplots(3, 3, figsize=(15, 14))
for row, r in enumerate(reports):
    cands = all_candidates[r.name][:3]
    for col in range(3):
        ax = axs[row, col]
        if col >= len(cands):
            ax.axis('off')
            continue
        d = cands[col]
        z = d['frame']
        img = np.asarray(r.volume[z], float)
        ax.imshow(img, cmap='gray', vmin=-150, vmax=750)
        p = d['path']
        ax.plot(p[:, 1], p[:, 0], linewidth=1.8)
        plaque = (canonical[r.name].refined_mask[z] == 2) & d['tube']
        if np.any(plaque):
            overlay = np.ma.masked_where(~plaque, plaque)
            ax.imshow(overlay, cmap='autumn', alpha=.65, interpolation='nearest')
        ax.set_title(
            f"{r.name} candidate {col+1}: frame {z}\n"
            f"path {d['path_length_mm']:.1f} mm; HU {d['mean_path_hu']:.0f}; plaque {d['plaque_tube_voxels']} px"
        )
        ax.axis('off')
fig.suptitle('Image-driven coronary tracking candidates — path from CT image, not vessel mask', fontsize=15)
plt.tight_layout(rect=[0, 0, 1, .97])
plt.savefig(OUT / '01_tracking_candidates.png', dpi=180, bbox_inches='tight')
plt.show()
plt.close(fig)

# Selected view plus straightened vessel-centered strip.
fig, axs = plt.subplots(3, 2, figsize=(18, 14), gridspec_kw={'width_ratios': [1, 1.65]})
along_rows = []
for row, r in enumerate(reports):
    d = selected[r.name]
    z = d['frame']
    img = np.asarray(r.volume[z], float)
    plaque = (canonical[r.name].refined_mask[z] == 2) & d['tube']
    p = d['path']

    ax = axs[row, 0]
    ax.imshow(img, cmap='gray', vmin=-150, vmax=750)
    ax.plot(p[:, 1], p[:, 0], linewidth=2)
    if np.any(plaque):
        overlay = np.ma.masked_where(~plaque, plaque)
        ax.imshow(overlay, cmap='autumn', alpha=.65, interpolation='nearest')
    y0 = max(0, int(np.floor(p[:, 0].min())) - 45)
    y1 = min(img.shape[0], int(np.ceil(p[:, 0].max())) + 46)
    x0 = max(0, int(np.floor(p[:, 1].min())) - 45)
    x1 = min(img.shape[1], int(np.ceil(p[:, 1].max())) + 46)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_title(f"{r.name} selected CPR frame {z}\ntracked path + plaque within 5 mm")
    ax.axis('off')

    spacing = r.mask_image.GetSpacing()
    sp_yx = (float(spacing[1]), float(spacing[0]))
    st = straighten(img, plaque, p, sp_yx)
    ax2 = axs[row, 1]
    if st is None:
        ax2.text(.5, .5, 'Straightening failed', ha='center', va='center')
        ax2.axis('off')
        continue

    s, off, strip, plaque_strip = st
    ax2.imshow(strip, cmap='gray', vmin=-150, vmax=750, aspect='auto', origin='lower', extent=[s[0], s[-1], off[0], off[-1]])
    if np.any(plaque_strip):
        overlay = np.ma.masked_where(~plaque_strip, plaque_strip)
        ax2.imshow(overlay, cmap='autumn', alpha=.70, aspect='auto', origin='lower', extent=[s[0], s[-1], off[0], off[-1]], interpolation='nearest')
    ax2.axhline(0, linewidth=1)
    ax2.set_xlabel('Distance along tracked CPR path (mm)')
    ax2.set_ylabel('Transverse distance (mm)')
    ax2.set_title(f'{r.name} straightened vessel-centered display')

    bins = np.arange(0, max(1, math.ceil(s[-1])) + 1, 1.0)
    center_idx = int(np.argmin(np.abs(off)))
    for a, b in zip(bins[:-1], bins[1:]):
        jj = (s >= a) & (s < b)
        along_rows.append({
            'vessel': r.name,
            'start_mm': a,
            'end_mm': b,
            'display_plaque_pixels': int(plaque_strip[:, jj].sum()) if np.any(jj) else 0,
            'mean_centerline_hu': float(np.nanmean(strip[center_idx, jj])) if np.any(jj) else np.nan,
        })

fig.suptitle('OpenPlaque image-driven straightened coronary plaque roadmaps — visualization only', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, .97])
plt.savefig(OUT / '02_straightened_coronary_roadmaps.png', dpi=180, bbox_inches='tight')
plt.show()
plt.close(fig)

pd.DataFrame(along_rows).to_csv(OUT / 'plaque_along_tracked_path.csv', index=False)

# Reuse the already validated PCAT presentation from the previous report, when available.
previous_dirs = [
    ROOT / 'Coronary_CPR_Roadmap_Report',
    ROOT / 'User_Friendly_Plaque_PCAT_Report_v3',
    ROOT / 'User_Friendly_Plaque_PCAT_Report_v2',
]
pcat_cross = None
pcat_ribbon = None
for d in previous_dirs:
    for name in ('02_rca_pcat_cross_sections.png', '02_rca_pcat_cross_sections_v3.png', '02_rca_pcat_cross_sections_v2.png'):
        p = d / name
        if p.exists() and pcat_cross is None:
            pcat_cross = p
    for name in ('03_rca_longitudinal_pcat_ribbon.png', '03_rca_longitudinal_pcat_ribbon_v3.png', '03_rca_longitudinal_pcat_ribbon_v2.png'):
        p = d / name
        if p.exists() and pcat_ribbon is None:
            pcat_ribbon = p

if pcat_cross is not None:
    shutil.copyfile(pcat_cross, OUT / '03_rca_pcat_cross_sections.png')
if pcat_ribbon is not None:
    shutil.copyfile(pcat_ribbon, OUT / '04_rca_longitudinal_pcat_ribbon.png')

metrics_dir = ROOT / 'Combined_TPV_PCAT_All_Metrics_v2'
tpv = pd.read_csv(metrics_dir / 'tpv_metrics_by_vessel_v2.csv')
pcat = pd.read_csv(metrics_dir / 'pcat_canonical_primary_v2.csv')
ps = pd.read_csv(metrics_dir / 'pcat_circular_sensitivity_v2.csv')
total = tpv[tpv.vessel == 'TOTAL'].iloc[0]
primary = pcat.iloc[0]
cmin = float(ps.pcat_mean_hu.min())
cmax = float(ps.pcat_mean_hu.max())
directional = -94.345186
fmin = min(cmin, directional)
fmax = max(cmax, directional)

fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
ax1 = fig.add_subplot(gs[0, 0])
vv = tpv[tpv.vessel != 'TOTAL']
ax1.bar(vv.vessel, vv.canonical_refined_tpv_mm3)
ax1.set_ylabel('Refined TPV (mm³)')
ax1.set_title('Canonical plaque volume by artery')

ax2 = fig.add_subplot(gs[0, 1])
ax2.axis('off')
ax2.text(
    .03, .95,
    f"Total refined TPV: {total.canonical_refined_tpv_mm3:.0f} mm³\n"
    f"Raw TPV: {total.raw_tpv_mm3:.0f} mm³\n"
    f"TPV sensitivity: {total.sensitivity_min_mm3:.0f}–{total.sensitivity_max_mm3:.0f} mm³\n\n"
    f"RCA 10–50 mm PCAT: {primary.pcat_mean_hu:.2f} HU\n"
    f"Circular-margin range: {cmin:.2f} to {cmax:.2f} HU\n"
    f"Full tested geometry: {fmin:.2f} to {fmax:.2f} HU",
    va='top', fontsize=13,
)

ax3 = fig.add_subplot(gs[1, :])
ax3.axis('off')
cols = ['vessel', 'selected_frame', 'path_length_mm', 'mean_path_hu', 'plaque_voxels_within_5mm_tube', 'tracking_status']
table_df = tracking_qc[cols].copy()
table_df['path_length_mm'] = table_df['path_length_mm'].map(lambda x: f'{x:.1f}')
table_df['mean_path_hu'] = table_df['mean_path_hu'].map(lambda x: f'{x:.0f}')
t = ax3.table(
    cellText=table_df.values,
    colLabels=['Vessel', 'CPR frame', 'Tracked length mm', 'Mean path HU', 'Displayed plaque px', 'QC'],
    loc='center', cellLoc='center',
)
t.auto_set_font_size(False)
t.set_fontsize(11)
t.scale(1, 1.6)
ax3.set_title('Image-driven CPR tracking QC — display only', pad=12)
fig.suptitle('OpenPlaque summary: quantitative endpoints + image-driven visualization QC', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, .96])
plt.savefig(OUT / '05_summary_dashboard.png', dpi=180, bbox_inches='tight')
plt.show()
plt.close(fig)

summary = pd.DataFrame([{
    'canonical_total_tpv_mm3': float(total.canonical_refined_tpv_mm3),
    'raw_total_tpv_mm3': float(total.raw_tpv_mm3),
    'tpv_sensitivity_min_mm3': float(total.sensitivity_min_mm3),
    'tpv_sensitivity_max_mm3': float(total.sensitivity_max_mm3),
    'rca_pcat_mean_hu': float(primary.pcat_mean_hu),
    'pcat_circular_min_hu': cmin,
    'pcat_circular_max_hu': cmax,
    'pcat_full_tested_min_hu': fmin,
    'pcat_full_tested_max_hu': fmax,
    'tracking_all_ok': bool((tracking_qc.tracking_status == 'OK').all()),
}])
summary.to_csv(OUT / 'tracking_report_summary.csv', index=False)


def img_b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()

sections = []
for fn, title in [
    ('01_tracking_candidates.png', 'Image-driven tracking candidates'),
    ('02_straightened_coronary_roadmaps.png', 'Straightened plaque roadmaps'),
    ('03_rca_pcat_cross_sections.png', 'RCA PCAT cross-sections'),
    ('04_rca_longitudinal_pcat_ribbon.png', 'RCA longitudinal PCAT'),
    ('05_summary_dashboard.png', 'Summary dashboard'),
]:
    p = OUT / fn
    if p.exists():
        sections.append("<h2>" + title + "</h2><img src='data:image/png;base64," + img_b64(p) + "' style='max-width:100%;height:auto'>")

html = (
    "<!doctype html><html><head><meta charset='utf-8'><title>OpenPlaque image-driven coronary tracking</title>"
    "<style>body{font-family:Arial,sans-serif;max-width:1400px;margin:30px auto;line-height:1.4}"
    "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}.note{background:#fff3cd;padding:12px}</style>"
    "</head><body><h1>OpenPlaque — Image-Driven Coronary Tracking + PCAT</h1>"
    "<div class='note'><b>Research visualization.</b> Canonical TPV is unchanged. The straightened plaque maps are derived from one Siemens CPR rotation per vessel and are not source-volume co-registration. RCA PCAT is a separate source-volume measurement and is an imaging surrogate related to perivascular inflammation, not a direct inflammation measurement or Caristo FAI-Score.</div>"
    "<h2>Tracking QC</h2>" + tracking_qc.to_html(index=False, float_format=lambda x: f'{x:.3f}') + ''.join(sections) + "</body></html>"
)
(OUT / 'OPENPLAQUE_IMAGE_DRIVEN_TRACKING_REPORT.html').write_text(html)

files = [
    'tracking_qc.csv',
    'plaque_along_tracked_path.csv',
    'tracking_report_summary.csv',
    '01_tracking_candidates.png',
    '02_straightened_coronary_roadmaps.png',
    '03_rca_pcat_cross_sections.png',
    '04_rca_longitudinal_pcat_ribbon.png',
    '05_summary_dashboard.png',
    'OPENPLAQUE_IMAGE_DRIVEN_TRACKING_REPORT.html',
]
zip_path = OUT / 'OPENPLAQUE_IMAGE_DRIVEN_TRACKING_REPORT_BACK.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for fn in files:
        p = OUT / fn
        if p.exists():
            z.write(p, arcname=fn)

print('REPORT BACK ZIP:', zip_path)
print('Drive search: https://drive.google.com/drive/u/0/search?q=OPENPLAQUE_IMAGE_DRIVEN_TRACKING_REPORT_BACK.zip')
