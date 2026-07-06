import numpy as np
from pathlib import Path
import tempfile
import pytest
sitk = pytest.importorskip("SimpleITK")

from openplaque.boundary import refine_plaque_mask
from openplaque.boundary_parameter_tuning import (
    collect_sample_cases,
    parameter_grid_dataframe,
    parameter_grid_size,
    evaluate_all_cases,
    aggregate_by_params,
    select_best_params,
)


def _write_nii(path, arr):
    img = sitk.GetImageFromArray(arr.astype(np.float32 if arr.dtype.kind == 'f' else np.uint8))
    img.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(img, str(path))


def test_refine_added_parameters_smoke():
    vol = np.zeros((8, 8, 8), dtype=np.float32)
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[2:5, 2:5, 2:5] = 2
    out = refine_plaque_mask(vol, mask, (1, 1, 1), closing_radius_voxels=1, fill_holes=True, connectivity=26)
    assert out.refined_mask.shape == mask.shape
    assert out.refined_plaque_voxels > 0


def test_parameter_tuning_end_to_end_with_cached_prediction():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        image = np.zeros((8, 8, 8), dtype=np.float32)
        label = np.zeros((8, 8, 8), dtype=np.uint8)
        pred = np.zeros((8, 8, 8), dtype=np.uint8)
        image[2:5, 2:5, 2:5] = 100
        label[2:5, 2:5, 2:5] = 2
        pred[2:5, 2:5, 2:5] = 2
        pred[0, 0, 0] = 2
        _write_nii(root / 'P01_LAD_axial_0000.nii.gz', image)
        _write_nii(root / 'P01_LAD_axial.nii.gz', label)
        pred_dir = root / 'preds'
        pred_dir.mkdir()
        _write_nii(pred_dir / 'P01_LAD_axial.nii.gz', pred)

        cases = collect_sample_cases(root)
        grid = {
            'min_component_voxels': [1, 5],
            'lumen_distance_voxels': [0],
            'high_hu_threshold': [None],
            'low_hu_threshold': [None],
            'closing_radius_voxels': [0],
            'fill_holes': [False],
            'connectivity': [26],
            'erode_core': [False],
            'erosion_iterations': [1],
        }
        assert parameter_grid_size(grid) == 2
        assert len(parameter_grid_dataframe(grid)) == 2
        rows = evaluate_all_cases(cases, pred_dir, grid)
        summary = aggregate_by_params(rows)
        params = select_best_params(rows)
        assert not rows.empty
        assert not summary.empty
        assert params['min_component_voxels'] in [1, 5]

from openplaque.boundary_parameter_tuning import archive_prediction_cache, restore_prediction_cache_from_archive


def test_prediction_archive_round_trip():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        pred_dir = root / 'preds'
        pred_dir.mkdir()
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[1:3, 1:3, 1:3] = 2
        _write_nii(pred_dir / 'P01_LAD_axial.nii.gz', arr)
        archive = root / 'prediction_cache.zip'
        archive_prediction_cache(pred_dir, archive)
        assert archive.exists()
        restored = root / 'restored'
        restore_prediction_cache_from_archive(archive, restored)
        assert (restored / 'P01_LAD_axial.nii.gz').exists()
