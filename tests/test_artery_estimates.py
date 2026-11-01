import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import SimpleITK as sitk
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from openplaque.artery_estimates import (
    ArteryEstimateStudy, candidate_shell, extract_verified_zip, load_cached_mask,
    save_cached_mask, select_artery_series, summarize_artery, total_row,
    validate_geometry, validate_model,
)


def report(values, mask, spacing=(1., 1., 1.)):
    values = np.asarray(values, dtype=float).reshape(1, 1, -1)
    mask = np.asarray(mask, dtype=np.uint8).reshape(values.shape)
    image = sitk.GetImageFromArray(values)
    image.SetSpacing(spacing)
    segmentation = sitk.GetImageFromArray(mask)
    segmentation.CopyInformation(image)
    return SimpleNamespace(name='LAD', image=image, mask_image=segmentation, volume=values, mask=mask)


def test_bins_partition_core_and_candidates_without_merging():
    r = report([-31, -30, 29, 30, 129, 130, 349, 350, np.nan, np.inf], [2] * 10)
    row = summarize_artery(r)
    assert [row[f'model_{k}_voxels'] for k in ('low_attenuation', 'noncalcified_range', 'intermediate', 'high_attenuation')] == [2, 2, 2, 1]
    assert row['model_below_range_voxels'] == 1
    assert row['model_nonfinite_voxels'] == 2
    assert row['model_region_voxels'] == 10
    assert row['candidate_region_voxels'] == 0


def test_no_core_does_not_promote_candidates_to_plaque():
    row = summarize_artery(report([100, 350, 0], [1, 0, 0]), radius_mm=1)
    assert row['model_region_mm3'] == 0
    assert row['model_high_attenuation_fraction'] is None
    assert row['candidate_high_attenuation_voxels'] == 1


def test_empty_segmentation_has_no_phantom_shell():
    assert not candidate_shell(np.zeros((3, 3, 3)), (1, 1, 1), 10).any()


def test_radius_uses_physical_distance_and_xyz_to_zyx_order():
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[2, 2, 2] = 1
    shell = candidate_shell(mask, (0.5, 1, 2), 1)
    assert shell[2, 2, 4] and shell[2, 3, 2]
    assert not shell[3, 2, 2] and not shell[2, 3, 3]
    assert not candidate_shell(mask, (1, 1, 1), 0).any()


def test_total_fractions_weight_volume_not_voxel_counts():
    a = summarize_artery(report([0], [2], (1, 1, 1)))
    b = summarize_artery(report([500], [2], (2, 2, 2)))
    total = total_row([a, b])
    assert total['model_low_attenuation_fraction'] == pytest.approx(1 / 9)
    assert total['model_high_attenuation_fraction'] == pytest.approx(8 / 9)
    assert total['model_region_mm3'] == 9
    assert total['voxel_volume_mm3'] is None


@pytest.mark.parametrize('property,value', [('Origin', (1, 0, 0)), ('Spacing', (2, 1, 1)),
    ('Direction', (-1, 0, 0, 0, -1, 0, 0, 0, 1))])
def test_same_shape_geometry_mismatch_rejected(property, value):
    r = report([0], [2])
    getattr(r.mask_image, 'Set' + property)(value)
    with pytest.raises(ValueError, match='geometry mismatch'):
        summarize_artery(r)


def test_cache_requires_identity_checksum_and_geometry(tmp_path):
    r = report([100], [2])
    path = tmp_path / 'LAD.nii.gz'
    identity = {'study': 'A', 'series': 'one', 'model': 'version1'}
    save_cached_mask(path, r.image, r.mask_image, identity)
    assert load_cached_mask(path, r.image, identity)[1].item() == 2
    with pytest.raises(ValueError, match='identity'):
        load_cached_mask(path, r.image, dict(identity, study='B'))
    with open(path, 'ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        load_cached_mask(path, r.image, identity)
    Path(str(path) + '.json').unlink()
    with pytest.raises(ValueError, match='provenance'):
        load_cached_mask(path, r.image, identity)


def test_archive_identity_and_incomplete_cache(tmp_path):
    archive = tmp_path / 'study.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('image.txt', 'A')
    first, _ = extract_verified_zip(archive, tmp_path / 'cache')
    assert extract_verified_zip(archive, tmp_path / 'cache')[0] == first
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('image.txt', 'B')
    second, _ = extract_verified_zip(archive, tmp_path / 'cache')
    assert first != second
    assert (second / 'image.txt').read_text() == 'B'
    (second / '.openplaque-complete').unlink()
    with pytest.raises(ValueError, match='Incomplete'):
        extract_verified_zip(archive, tmp_path / 'cache')


def test_unsafe_zip_rejected(tmp_path):
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('../escape.txt', 'bad')
    with pytest.raises(ValueError, match='Unsafe'):
        extract_verified_zip(archive, tmp_path / 'cache')


def mock_study(records):
    study = ArteryEstimateStudy.__new__(ArteryEstimateStudy)
    study.series = records
    return study


def test_detection_rejects_ties_missing_fallback_and_duplicate_series():
    study = mock_study([{'series_number': i, 'uid': str(i), 'description': 'LAD CPR'} for i in (1, 2)])
    with pytest.raises(ValueError, match='Ambiguous'):
        select_artery_series(study, ['LAD'])
    assert select_artery_series(study, ['LAD'], {'LAD': 2}) == {'LAD': 2}
    with pytest.raises(ValueError, match='No confident'):
        select_artery_series(study, ['RCA'])
    with pytest.raises(ValueError, match='expected one UID'):
        select_artery_series(study, ['LAD'], {'LAD': 1035})
    with pytest.raises(ValueError, match='multiple arteries'):
        select_artery_series(study, ['LAD', 'RCA'], {'LAD': 1, 'RCA': 1})


def test_actual_dicom_shared_folder_selects_requested_uid(tmp_path):
    files = []
    for number, intensity in [(1, 100), (2, 500)]:
        uid = generate_uid()
        for z in range(2):
            path = tmp_path / f'{number}_{z}.dcm'
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = CTImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
            ds.SOPClassUID, ds.SOPInstanceUID = CTImageStorage, meta.MediaStorageSOPInstanceUID
            ds.SeriesInstanceUID, ds.StudyInstanceUID = uid, '1.2.3'
            ds.SeriesNumber, ds.SeriesDescription = number, 'LAD CPR'
            ds.Modality, ds.PatientID = 'CT', 'synthetic'
            ds.Rows = ds.Columns = 2
            ds.ImagePositionPatient = [0, 0, z]
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.PixelSpacing = [1, 1]
            ds.SliceThickness = 1
            ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit, ds.PixelRepresentation = 15, 1
            ds.RescaleSlope, ds.RescaleIntercept = 1, 0
            ds.PixelData = np.full((2, 2), intensity, dtype=np.int16).tobytes()
            ds.save_as(path, enforce_file_format=True)
            files.append(path)
    archive = tmp_path / 'dicom.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for path in files:
            z.write(path, 'shared/' + path.name)
    study = ArteryEstimateStudy(archive, tmp_path / 'cache')
    assert np.all(study.load_series(2)[1] == 500)
    assert np.all(study.load_series(1)[1] == 100)


def test_model_checks_all_requested_folds(tmp_path):
    folder = tmp_path / 'Dataset001_CCTA_DHM/nnUNetTrainer__nnUNetPlans__3d_fullres'
    folder.mkdir(parents=True)
    (folder / 'dataset.json').write_text(json.dumps({'labels': {'background': 0, 'vessel': 1, 'plaque': 2}}))
    (folder / 'plans.json').write_text('{}')
    for f in range(4):
        (folder / f'fold_{f}').mkdir()
        (folder / f'fold_{f}/checkpoint_final.pth').write_bytes(b'fixture')
    with pytest.raises(FileNotFoundError, match='fold_4'):
        validate_model(tmp_path)
    assert len(validate_model(tmp_path, folds=(0, 1))) == 4


@pytest.mark.parametrize('edges', [(0, 0, 30, 350), (0, 30, 130, np.nan)])
def test_invalid_thresholds_rejected(edges):
    with pytest.raises(ValueError, match='HU edges'):
        summarize_artery(report([0], [2]), edges=edges)


def test_notebook_measurement_and_exports_with_synthetic_reports(tmp_path):
    import sys
    import subprocess
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    notebook = Path(__file__).parents[1] / 'notebooks/13_Plaque_Labels_And_Candidates_Audited.ipynb'
    nb = json.loads(notebook.read_text())
    for index, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'cell_{index}', 'exec')
    plots = tmp_path / 'plots'
    plots.mkdir()
    context = dict(reports=[report([0, 100, 350], [2, 1, 0])],
        VESSEL_RADIUS_MM=1, HU_EDGES=(-20., 20., 120., 300.),
        pd=pd, subprocess=subprocess, sys=sys, json=json,
        OUTPUT_DIR=tmp_path, PLOT_DIR=plots, run_id='synthetic', commit='test',
        study=SimpleNamespace(study_sha256='test'), model_identity={},
        provenance=[], SERIES_OVERRIDES={}, display=lambda x: None)
    exec(''.join(nb['cells'][12]['source']), context)
    exec(''.join(nb['cells'][14]['source']), context)
    html = context['html_path'].read_text(encoding='utf-8')
    assert 'data:image/png;base64,' in html
    assert '-20 to &lt;20 HU' in html
    assert 'Unconfirmed candidates' in html
    assert 'mixed_intermediate' not in html
    assert (tmp_path / 'manifest.json').is_file()
    assert context['estimate_df'].iloc[-1]['model_region_mm3'] == 1
