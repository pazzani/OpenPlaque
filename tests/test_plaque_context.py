import csv

import numpy as np

from openplaque.plaque_context import (
    compute_hu_histogram,
    compute_plaque_context,
    dilate_plaque_mask,
    plaque_shell_mask,
    save_context_csv,
    summarize_hu_bins,
    write_context_html_report,
)


def test_summarize_hu_bins_requested_ranges():
    values = np.array([-10, 30, 129, 130, 349, 350, 699, 700, 1000, 1001], dtype=float)
    summary = summarize_hu_bins(values)

    assert summary["lt_30_voxels"] == 1
    assert summary["hu_30_130_voxels"] == 2
    assert summary["hu_130_350_voxels"] == 2
    assert summary["hu_350_700_voxels"] == 2
    assert summary["hu_700_1000_voxels"] == 2
    assert summary["gt_1000_voxels"] == 1


def test_dilate_and_shell_masks_one_voxel_3d():
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[2, 2, 2] = 2

    dilated = dilate_plaque_mask(mask, radius_voxels=1, connectivity=6)
    shell = plaque_shell_mask(mask, radius_voxels=1, connectivity=6)

    assert int(dilated.sum()) == 7
    assert int(shell.sum()) == 6
    assert not shell[2, 2, 2]


def test_compute_hu_histogram_inside_plaque_mask():
    volume = np.array([[[10, 50, 200, 500, 1200]]], dtype=float)
    mask = np.array([[[2, 2, 0, 2, 0]]], dtype=np.uint8)

    counts, edges = compute_hu_histogram(volume, mask, bins=[0, 30, 130, 700])

    assert edges.tolist() == [0, 30, 130, 700]
    assert counts.tolist() == [1, 1, 1]


def test_compute_plaque_context_detects_noncalcified_shell():
    volume = np.zeros((5, 5, 5), dtype=float)
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[2, 2, 2] = 2
    volume[2, 2, 2] = 600

    shell = plaque_shell_mask(mask, radius_voxels=1, connectivity=6)
    volume[shell] = 80

    rows = compute_plaque_context("LAD", volume, mask, radii_voxels=(1,), connectivity=6)
    plaque_row, shell_row = rows

    assert plaque_row["region"] == "plaque"
    assert plaque_row["hu_350_700_voxels"] == 1
    assert plaque_row["calcified_core_fraction"] == 1.0
    assert shell_row["region"] == "shell_1vox"
    assert shell_row["hu_30_130_voxels"] == 6
    assert shell_row["noncalcified_context_fraction"] == 1.0


def test_context_csv_and_html_exports(tmp_path):
    rows = [
        {
            "vessel": "RCA",
            "region": "plaque",
            "radius_voxels": 0,
            "voxel_count": 1,
            "mean_hu": 500.0,
        }
    ]

    csv_path = save_context_csv(rows, tmp_path / "context.csv")
    html_path = write_context_html_report(tmp_path / "context.html", rows)

    with csv_path.open(newline="", encoding="utf-8") as f:
        loaded = list(csv.DictReader(f))
    assert loaded[0]["vessel"] == "RCA"
    assert "OpenPlaque Plaque HU Context Report" in html_path.read_text(encoding="utf-8")
