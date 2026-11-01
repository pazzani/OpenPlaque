"""Auditable input handling and HU summaries for notebook 13 only.

HU candidates are descriptive search-region measurements, never added to plaque.
The legacy study/detection/reporting pipelines retain their existing behavior.
"""
from pathlib import Path
import hashlib
import json
import tempfile
import zipfile

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from .study import OpenPlaqueStudy
from .artery_detection import score_series_for_artery

HU_KEYS = ('low_attenuation', 'noncalcified_range', 'intermediate', 'high_attenuation')
DEFAULT_EDGES = (-30.0, 30.0, 130.0, 350.0)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def extract_verified_zip(zip_path, cache_root):
    """Content-addressed, atomic extraction; never trust a pre-existing folder."""
    digest = file_sha256(zip_path)
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / digest
    marker = target / '.openplaque-complete'
    if marker.is_file() and marker.read_text() == digest:
        return target, digest
    if target.exists():
        raise ValueError(f'Incomplete extraction cache: {target}; use a new cache root.')
    with tempfile.TemporaryDirectory(dir=root) as temp:
        staging = Path(temp) / 'data'
        staging.mkdir()
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                member = Path(info.filename)
                if member.is_absolute() or '..' in member.parts or '\\' in info.filename or ':' in info.filename:
                    raise ValueError(f'Unsafe ZIP member: {info.filename}')
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('ZIP symlinks are not supported')
            archive.extractall(staging)
        (staging / '.openplaque-complete').write_text(digest)
        staging.rename(target)
    return target, digest


class ArteryEstimateStudy(OpenPlaqueStudy):
    """Strict study loader used only by the corrected notebook."""
    def __init__(self, zip_path, cache_root='/content/openplaque_study_cache'):
        self.zip_path = str(zip_path)
        root, self.study_sha256 = extract_verified_zip(zip_path, cache_root)
        self.extract_root = str(root)
        self.series = []
        self._scan()

    def series_record(self, series_number):
        matches = [s for s in self.series if s['series_number'] == int(series_number)]
        if len(matches) != 1:
            raise ValueError(f'Series {series_number}: expected one UID, found {len(matches)}')
        return matches[0]

    def load_series(self, series_number):
        record = self.series_record(series_number)
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(record['folder'], record['uid'])
        if len(files) != record['images']:
            raise ValueError(f"Incomplete series {record['uid']}: expected {record['images']} images, found {len(files)}")
        reader.SetFileNames(files)
        image = reader.Execute()
        return image, sitk.GetArrayFromImage(image), files


def select_artery_series(study, vessels=('LAD', 'RCA', 'LCX'), overrides=None):
    """Reject ties, weak matches, duplicate UIDs and absent explicit selections."""
    overrides = overrides or {}
    if set(overrides) - set(vessels):
        raise ValueError('Unknown artery in series overrides')
    selected = {}
    used = set()
    for vessel in vessels:
        if vessel in overrides:
            number = int(overrides[vessel])
        else:
            ranked = sorted((score_series_for_artery(r, vessel) for r in study.series),
                            key=lambda c: c.score, reverse=True)
            if not ranked or ranked[0].score < 100:
                raise ValueError(f'No confident {vessel} series; inspect study.summary() and set SERIES_OVERRIDES.')
            if len(ranked) > 1 and ranked[0].score == ranked[1].score:
                raise ValueError(f'Ambiguous {vessel} series; set SERIES_OVERRIDES after inspection.')
            number = ranked[0].series_number
        record = study.series_record(number)
        if record['uid'] in used:
            raise ValueError('The same DICOM series cannot represent multiple arteries')
        used.add(record['uid'])
        selected[vessel] = number
    return selected


def geometry(image):
    return {key: list(getattr(image, method)()) for key, method in
            [('size', 'GetSize'), ('spacing', 'GetSpacing'),
             ('origin', 'GetOrigin'), ('direction', 'GetDirection')]}


def validate_geometry(image, mask_image):
    if image.GetDimension() != 3 or mask_image.GetDimension() != 3:
        raise ValueError('Expected 3-D image and mask')
    a, b = geometry(image), geometry(mask_image)
    if a['size'] != b['size'] or any(not np.allclose(a[k], b[k], rtol=0, atol=1e-5)
                                               for k in ('spacing', 'origin', 'direction')):
        raise ValueError('Image/mask geometry mismatch; regenerate the segmentation')
    spacing = np.asarray(a['spacing'])
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('Invalid physical spacing')
    direction = np.asarray(a['direction']).reshape(3, 3)
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError('Physical-distance dilation requires orthonormal image directions')


def load_cached_mask(path, image, identity):
    path = Path(path)
    sidecar = Path(str(path) + '.json')
    if not path.is_file() or not sidecar.is_file():
        raise ValueError('Mask cache lacks provenance')
    record = json.loads(sidecar.read_text())
    if record.get('identity') != identity or record.get('sha256') != file_sha256(path):
        raise ValueError('Mask cache identity or checksum mismatch')
    mask_image = sitk.ReadImage(str(path))
    validate_geometry(image, mask_image)
    return mask_image, sitk.GetArrayFromImage(mask_image)


def save_cached_mask(path, image, mask_image, identity):
    validate_geometry(image, mask_image)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(mask_image, str(path))
    Path(str(path) + '.json').write_text(json.dumps({
        'identity': identity, 'sha256': file_sha256(path), 'geometry': geometry(image)
    }, indent=2))


def validate_model(results_root, dataset='Dataset001_CCTA_DHM', folds=(0, 1, 2, 3, 4)):
    """Check the exact default trainer/plans/config used by run_nnunet."""
    folder = Path(results_root) / dataset / 'nnUNetTrainer__nnUNetPlans__3d_fullres'
    paths = [folder / 'dataset.json', folder / 'plans.json']
    paths += [folder / f'fold_{f}' / 'checkpoint_final.pth' for f in folds]
    missing = [str(p) for p in paths if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise FileNotFoundError('Missing nnU-Net model files: ' + ', '.join(missing))
    labels = json.loads(paths[0].read_text()).get('labels', {})
    if any(not isinstance(v, int) for v in labels.values()) or set(labels.values()) != {0, 1, 2}:
        raise ValueError('Expected model labels 0=background, 1=vessel, 2=plaque; inspect dataset.json')
    names = {v: k.lower().replace('_', ' ').strip() for k, v in labels.items()}
    if names[0] != 'background' or names[1] not in ('vessel', 'normal vessel', 'lumen', 'coronary', 'coronary artery') or names[2] != 'plaque':
        raise ValueError(f'Unrecognized model label semantics: {labels}; verify the model before use')
    return {str(p.relative_to(results_root)): file_sha256(p) for p in paths}


def candidate_shell(mask, spacing_xyz, radius_mm):
    spacing = np.asarray(spacing_xyz, dtype=float)
    if spacing.shape != (3,) or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('Expected three positive finite spacing values')
    if not np.isfinite(radius_mm) or radius_mm < 0:
        raise ValueError('radius_mm must be finite and nonnegative')
    anatomy = (mask == 1) | (mask == 2)
    if not anatomy.any() or radius_mm == 0:
        return np.zeros(mask.shape, dtype=bool)
    distance = ndi.distance_transform_edt(~anatomy, sampling=spacing[::-1])
    return (~anatomy) & (distance <= radius_mm)


def summarize_artery(report, radius_mm=0.5, edges=DEFAULT_EDGES):
    """Report HU bins separately for model label 2 and unconfirmed outer shell."""
    validate_geometry(report.image, report.mask_image)
    volume, mask = np.asarray(report.volume), np.asarray(report.mask)
    if volume.shape != mask.shape or volume.shape != tuple(reversed(report.image.GetSize())):
        raise ValueError('Image, volume and mask shapes differ')
    if not np.isin(mask, [0, 1, 2]).all():
        raise ValueError('Unexpected segmentation labels')
    edges = np.asarray(edges, dtype=float)
    if edges.shape != (4,) or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
        raise ValueError('HU edges must contain four strictly increasing finite values')
    spacing = report.image.GetSpacing()
    voxel_volume = float(np.prod(spacing))
    shell = candidate_shell(mask, spacing, radius_mm)
    row = {'artery': report.name, 'voxel_volume_mm3': voxel_volume}
    for region, selection in [('model', mask == 2), ('candidate', shell)]:
        values = volume[selection]
        finite = np.isfinite(values)
        counts = {}
        for index, key in enumerate(HU_KEYS):
            selected = finite & (values >= edges[index])
            if index < 3:
                selected &= values < edges[index + 1]
            counts[key] = int(selected.sum())
        counts['below_range'] = int((finite & (values < edges[0])).sum())
        counts['nonfinite'] = int((~finite).sum())
        counts['classified'] = sum(counts[k] for k in HU_KEYS)
        counts['region'] = int(values.size)
        for key, count in counts.items():
            row[f'{region}_{key}_voxels'] = count
            row[f'{region}_{key}_mm3'] = count * voxel_volume
        for key in HU_KEYS:
            row[f'{region}_{key}_fraction'] = counts[key] / counts['classified'] if counts['classified'] else None
    return row


def total_row(rows):
    if not rows:
        raise ValueError('No artery rows')
    total = {'artery': 'TOTAL', 'voxel_volume_mm3': None}
    for key in rows[0]:
        if key != 'voxel_volume_mm3' and key.endswith(('_voxels', '_mm3')):
            total[key] = sum(row[key] for row in rows)
    for region in ('model', 'candidate'):
        denominator = total[f'{region}_classified_mm3']
        for key in HU_KEYS:
            total[f'{region}_{key}_fraction'] = total[f'{region}_{key}_mm3'] / denominator if denominator else None
    return total
