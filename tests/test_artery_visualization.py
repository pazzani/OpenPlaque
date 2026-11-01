import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk

from openplaque.artery_visualization import (
    artery_3d, artery_tabs, initial_slice, plaque_classes, save_overview,
    slice_data, surface_mesh,
)


def make_report(name='LAD', empty=False):
    mask = np.zeros((12, 28, 28), dtype=np.uint8)
    volume = np.full(mask.shape, -150., dtype=float)
    yy, xx = np.indices(mask.shape[1:])
    if not empty:
        for z in range(12):
            cx = 12 + z // 4
            vessel = (xx-cx)**2 + (yy-14)**2 <= 16
            plaque = ((xx-cx)**2 + (yy-14)**2 <= 36) & ~vessel & (xx > cx)
            mask[z, vessel] = 1
            volume[z, vessel] = 400
            mask[z, plaque] = 2
            volume[z, plaque] = [0, 100, 200, 500][z // 3]
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing((0.4, 0.6, 1.2))
    image.SetOrigin((20, 30, 40))
    image.SetDirection((0, -1, 0, 1, 0, 0, 0, 0, 1))
    mask_image = sitk.GetImageFromArray(mask)
    mask_image.CopyInformation(image)
    return SimpleNamespace(name=name, image=image, mask_image=mask_image, volume=volume, mask=mask)


def test_color_bins_only_apply_to_model_plaque_and_preserve_unclassified():
    values = np.array([-40, -30, 30, 130, 350, np.nan, 500, 500])
    mask = np.array([2, 2, 2, 2, 2, 2, 1, 0])
    assert plaque_classes(values, mask).tolist() == [5, 1, 2, 3, 4, 5, 0, 0]


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_slice_extent_and_overlay_alignment(axis):
    report = make_report()
    index = initial_slice(report.mask, axis)
    ct, rgba, extent = slice_data(report, axis, index)
    expected_mask = np.take(report.mask, index, axis)
    assert rgba.shape == ct.shape + (4,)
    assert np.all(rgba[expected_mask == 0, 3] == 0)
    assert np.all(rgba[expected_mask == 2, 3] == 0.75)
    row, col = np.delete([1.2, 0.6, 0.4], axis)
    assert extent == (0, ct.shape[1]*col, ct.shape[0]*row, 0)
    _, hidden, _ = slice_data(report, axis, index, show_plaque=False, show_vessel=False)
    assert not hidden.any()


def test_mesh_preserves_physical_coordinates_and_single_voxel():
    report = make_report(empty=True)
    selected = np.zeros(report.mask.shape, dtype=bool)
    selected[2, 3, 4] = True
    points, faces = surface_mesh(selected, report.image)
    spacing = np.array(report.image.GetSpacing())
    direction = np.array(report.image.GetDirection()).reshape(3, 3)
    center = direction @ (np.array([4, 3, 2]) * spacing) + report.image.GetOrigin()
    radius = np.abs(direction) @ (spacing / 2)
    assert len(faces) > 0
    assert points.min(axis=0) == pytest.approx(center-radius)
    assert points.max(axis=0) == pytest.approx(center+radius)
    coarse_points, coarse_faces = surface_mesh(selected, report.image, step_size=4)
    assert len(coarse_faces) > 0  # Tiny isolated objects get a native-grid fallback.


def test_3d_classes_do_not_change_inputs_or_include_background():
    report = make_report()
    original_mask, original_volume = report.mask.copy(), report.volume.copy()
    figure = artery_3d(report)
    assert [trace.name for trace in figure.data] == [
        'Vessel (label 1)', 'Low attenuation', 'Noncalcified', 'Mixed/intermediate', 'Calcified']
    assert [trace.color for trace in figure.data][1:] == ['#d73027', '#fdae61', '#2b83ba', '#f7f7f7']
    assert np.array_equal(report.mask, original_mask)
    assert np.array_equal(report.volume, original_volume)


def test_empty_artery_remains_viewable(tmp_path):
    report = make_report(empty=True)
    assert initial_slice(report.mask) == report.mask.shape[0] // 2
    figure = artery_3d(report)
    assert not figure.data
    assert 'No segmented vessel or plaque' in figure.layout.title.text
    assert save_overview(report, tmp_path / 'empty.png').stat().st_size > 0


def test_independent_artery_tabs_and_plane_switching(monkeypatch):
    import IPython.display
    monkeypatch.setattr(IPython.display, 'display', lambda *args, **kwargs: None)
    tabs = artery_tabs([make_report('LAD'), make_report('RCA'), make_report('LCX')])
    assert tabs.titles == ('LAD', 'RCA', 'LCX')
    first, second = tabs.children[:2]
    first.children[1].children[0].value = 'X slices'
    assert first.children[2].max == 27
    assert second.children[2].max == 11
    first.children[2].value = 27
    first.children[1].children[0].value = 'Z slices'
    assert 0 <= first.children[2].value <= 11
    tabs.close()


def test_notebook_exports_all_artery_views_and_links(tmp_path):
    import sys
    import subprocess
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    nb = json.loads((Path(__file__).parents[1] / 'notebooks/13_Plaque_Labels_And_Candidates_Audited.ipynb').read_text())
    plots = tmp_path / 'plots'
    plots.mkdir()
    context = dict(reports=[make_report(v) for v in ('LAD', 'RCA', 'LCX')],
                   VESSEL_RADIUS_MM=0.5, HU_EDGES=(-30, 30, 130, 350),
                   pd=pd, subprocess=subprocess, sys=sys, json=json, Path=Path,
                   OUTPUT_DIR=tmp_path, PLOT_DIR=plots, run_id='synthetic', commit='test',
                   study=SimpleNamespace(study_sha256='test'), model_identity={},
                   provenance=[], SERIES_OVERRIDES={}, display=lambda *args: None,
                   SHOW_INTERACTIVE_VIEWS=False, CREATE_3D_VIEWS=True, MESH_STEP_SIZE=1)
    exec(''.join(nb['cells'][12]['source']), context)
    viewer = next(c for c in nb['cells'] if c.get('id') == 'artery-visualization')
    exec(''.join(viewer['source']), context)
    exec(''.join(nb['cells'][-1]['source']), context)
    html = context['html_path'].read_text(encoding='utf-8')
    for vessel in ('LAD', 'RCA', 'LCX'):
        assert (plots / f'{vessel}_artery_plaque_overview.png').is_file()
        viewer_file = tmp_path / f'{vessel}_artery_plaque_3d.html'
        assert viewer_file.name in html
        contents = viewer_file.read_text(encoding='utf-8')
        assert 'Plotly.newPlot' in contents and 'plotly.js v' in contents
    assert len(json.loads((tmp_path / 'manifest.json').read_text())['visualization']['files']) == 3
