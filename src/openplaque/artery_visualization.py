"""Read-only artery and plaque viewers for the audited notebook."""
from pathlib import Path

import numpy as np

from .artery_estimates import DEFAULT_EDGES, validate_geometry

PLAQUE_LABELS = ('Low attenuation', 'Noncalcified', 'Mixed/intermediate', 'Calcified')
PLAQUE_COLORS = ('#d73027', '#fdae61', '#2b83ba', '#f7f7f7')
VESSEL_COLOR = '#00bcd4'
UNCLASSIFIED_COLOR = '#b358c8'
PLANES = {'Z slices': 0, 'Y slices': 1, 'X slices': 2}


def _validate(report, edges):
    validate_geometry(report.image, report.mask_image)
    if report.volume.shape != report.mask.shape or report.mask.shape != tuple(reversed(report.image.GetSize())):
        raise ValueError('Image and segmentation arrays must have matching shapes')
    if not np.isin(report.mask, (0, 1, 2)).all():
        raise ValueError('Expected labels 0, 1 and 2')
    edges = np.asarray(edges, dtype=float)
    if edges.shape != (4,) or not np.isfinite(edges).all() or not (np.diff(edges) > 0).all():
        raise ValueError('Expected four increasing finite HU edges')
    return edges


def plaque_classes(volume, mask, edges=DEFAULT_EDGES):
    """Display codes 1-4 for plaque HU bins, 5 for unclassified label-2 voxels."""
    edges = np.asarray(edges, dtype=float)
    if edges.shape != (4,) or not np.isfinite(edges).all() or not (np.diff(edges) > 0).all():
        raise ValueError('Expected four increasing finite HU edges')
    if volume.shape != mask.shape:
        raise ValueError('Image and mask shapes differ')
    classes = np.zeros(mask.shape, dtype=np.uint8)
    plaque = mask == 2
    classes[plaque] = 5
    finite = np.isfinite(volume)
    for i in range(4):
        selected = plaque & finite & (volume >= edges[i])
        if i < 3:
            selected &= volume < edges[i + 1]
        classes[selected] = i + 1
    return classes


def initial_slice(mask, axis=0):
    selection = mask == 2
    if not selection.any():
        selection = mask == 1
    if not selection.any():
        return mask.shape[axis] // 2
    other_axes = tuple(i for i in range(3) if i != axis)
    return int(np.argmax(selection.sum(axis=other_axes)))


def slice_data(report, axis, index, edges=DEFAULT_EDGES, show_vessel=True,
               show_plaque=True, color_types=True, opacity=0.75):
    """Return CT, transparent overlay and in-plane physical extent (millimeters)."""
    from matplotlib.colors import to_rgb

    if axis not in (0, 1, 2) or not 0 <= index < report.mask.shape[axis]:
        raise ValueError('Invalid slice selection')
    if not np.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError('Opacity must be between zero and one')
    ct = np.take(report.volume, index, axis=axis)
    mask = np.take(report.mask, index, axis=axis)
    overlay = np.zeros(mask.shape + (4,), dtype=float)
    if show_vessel:
        overlay[mask == 1] = (*to_rgb(VESSEL_COLOR), opacity * 0.35)
    if show_plaque:
        if color_types:
            classes = plaque_classes(ct, mask, edges)
            for code, color in enumerate((*PLAQUE_COLORS, UNCLASSIFIED_COLOR), start=1):
                overlay[classes == code] = (*to_rgb(color), opacity)
        else:
            overlay[mask == 2] = (*to_rgb('#ffcc00'), opacity)
    spacing = np.asarray(report.image.GetSpacing())[::-1]
    row_spacing, col_spacing = np.delete(spacing, axis)
    extent = (0, ct.shape[1] * col_spacing, ct.shape[0] * row_spacing, 0)
    return ct, overlay, extent


def _legend():
    from matplotlib.patches import Patch
    return [Patch(facecolor=VESSEL_COLOR, label='Vessel (label 1)')] + [
        Patch(facecolor=color, edgecolor='black', label=label)
        for label, color in zip((*PLAQUE_LABELS, 'Unclassified plaque'), (*PLAQUE_COLORS, UNCLASSIFIED_COLOR))
    ]


def slice_figure(report, axis=0, index=None, edges=DEFAULT_EDGES, center=350,
                 width=1100, opacity=0.75, show_vessel=True, show_plaque=True, color_types=True):
    import matplotlib.pyplot as plt

    if not np.isfinite(width) or width <= 0 or not np.isfinite(center):
        raise ValueError('Invalid CT window')
    index = initial_slice(report.mask, axis) if index is None else index
    ct, overlay, extent = slice_data(report, axis, index, edges, show_vessel, show_plaque, color_types, opacity)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax in axes:
        ax.imshow(ct, cmap='gray', vmin=center-width/2, vmax=center+width/2,
                  extent=extent, interpolation='nearest', aspect='equal')
        ax.set_xlabel('In-plane distance (mm)')
        ax.set_ylabel('In-plane distance (mm)')
    axes[0].set_title('CT')
    axes[1].imshow(overlay, extent=extent, interpolation='nearest', aspect='equal')
    axes[1].set_title('Vessel and plaque overlays' if color_types else 'Vessel and plaque (yellow)')
    fig.suptitle(f'{report.name} — image axis {"ZYX"[axis]}, slice {index}/{report.mask.shape[axis]-1}')
    if color_types:
        fig.legend(handles=_legend(), loc='lower center', ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0.12 if color_types else 0, 1, 0.95))
    return fig


def artery_tabs(reports, edges=DEFAULT_EDGES):
    """One notebook tab per artery; controls redraw slices without changing masks."""
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display, clear_output

    panels = []
    for report in reports:
        _validate(report, edges)
        def make_panel(report):
            plane = widgets.Dropdown(options=list(PLANES), description='Image axis')
            slider = widgets.IntSlider(min=0, max=report.mask.shape[0]-1,
                                       value=initial_slice(report.mask), description='Slice', continuous_update=False)
            center = widgets.IntSlider(min=-500, max=1500, value=350, step=25,
                                       description='Window center', continuous_update=False)
            width = widgets.IntSlider(min=100, max=3000, value=1100, step=50,
                                      description='Window width', continuous_update=False)
            opacity = widgets.FloatSlider(min=0, max=1, value=0.75, step=0.05,
                                          description='Opacity', continuous_update=False)
            for control in (slider, center, width, opacity):
                control.style.description_width = '110px'
                control.layout.width = '500px'
            vessel = widgets.Checkbox(value=True, description='Vessel')
            plaque = widgets.Checkbox(value=True, description='Plaque')
            types = widgets.Checkbox(value=True, description='Plaque colors')
            jump = widgets.Button(description='Largest plaque slice', layout=widgets.Layout(width='180px'))
            output = widgets.Output()
            busy = [False]

            def draw(change=None):
                if busy[0]:
                    return
                with output:
                    clear_output(wait=True)
                    fig = slice_figure(report, PLANES[plane.value], slider.value, edges,
                                       center.value, width.value, opacity.value,
                                       vessel.value, plaque.value, types.value)
                    display(fig)
                    plt.close(fig)

            def change_plane(change):
                busy[0] = True
                axis = PLANES[plane.value]
                slider.max = report.mask.shape[axis]-1
                slider.value = initial_slice(report.mask, axis)
                busy[0] = False
                draw()

            def jump_to_plaque(button):
                slider.value = initial_slice(report.mask, PLANES[plane.value])

            plane.observe(change_plane, names='value')
            for control in (slider, center, width, opacity, vessel, plaque, types):
                control.observe(draw, names='value')
            jump.on_click(jump_to_plaque)
            message = 'Use the slider to inspect the artery. Axes follow the supplied image grid.'
            if not np.any(report.mask == 2):
                message += ' No model-labeled plaque in this artery.'
            panel = widgets.VBox([widgets.HTML(message), widgets.HBox([plane, jump]), slider,
                                  widgets.HBox([vessel, plaque, types]), center, width, opacity, output])
            draw()
            return panel
        panels.append(make_panel(report))
    tabs = widgets.Tab(children=panels)
    for i, report in enumerate(reports):
        tabs.set_title(i, str(report.name))
    return tabs


def surface_mesh(selection, image, step_size=1):
    """Mesh a cropped binary mask, preserving spacing, origin and direction."""
    from skimage.measure import marching_cubes

    if step_size < 1 or int(step_size) != step_size:
        raise ValueError('Mesh step size must be a positive integer')
    if not np.any(selection):
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    bounds = []
    for axis in range(3):
        occupied = np.flatnonzero(selection.any(axis=tuple(i for i in range(3) if i != axis)))
        bounds.append((int(occupied[0]), int(occupied[-1])+1))
    crop = selection[tuple(slice(lo, hi) for lo, hi in bounds)]
    # Padding closes masks at image edges and also permits single-slice images.
    padded = np.pad(crop.astype(np.uint8), 1)
    try:
        vertices, faces, _, _ = marching_cubes(padded, level=0.5, step_size=int(step_size))
    except (ValueError, RuntimeError):
        if step_size != 1:
            # Coarse sampling may miss tiny plaque: retry at native resolution.
            vertices, faces, _, _ = marching_cubes(padded, level=0.5, step_size=1)
        else:
            raise
    vertices += np.array([lo for lo, _ in bounds]) - 1
    xyz = vertices[:, ::-1] * np.asarray(image.GetSpacing())
    xyz = xyz @ np.asarray(image.GetDirection()).reshape(3, 3).T + np.asarray(image.GetOrigin())
    return xyz, faces


def artery_3d(report, edges=DEFAULT_EDGES, step_size=1):
    """Rotatable model segmentation; candidates are deliberately not meshed as plaque."""
    import plotly.graph_objects as go

    _validate(report, edges)
    classes = plaque_classes(report.volume, report.mask, edges)
    fig = go.Figure()

    def add(selection, name, color, opacity):
        vertices, faces = surface_mesh(selection, report.image, step_size)
        if not len(faces):
            return
        fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                               i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], name=name,
                               color=color, opacity=opacity, showlegend=True, hoverinfo='name',
                               flatshading=False))
    add(report.mask == 1, 'Vessel (label 1)', VESSEL_COLOR, 0.18)
    for code, (label, color) in enumerate(zip((*PLAQUE_LABELS, 'Unclassified plaque'),
                                            (*PLAQUE_COLORS, UNCLASSIFIED_COLOR)), start=1):
        add(classes == code, label, color, 1.0)
    message = 'No model-labeled plaque' if not (report.mask == 2).any() else 'Model-labeled plaque'
    if not (report.mask > 0).any():
        message = 'No segmented vessel or plaque'
    fig.update_layout(template='plotly_dark', title=f'{report.name} — {message}<br><sup>Image-grid surfaces; mesh step {step_size}<br>Drag to rotate; scroll to zoom; click legend to hide/show</sup>',
                      scene=dict(aspectmode='data', xaxis_title='Physical X (mm)',
                                 yaxis_title='Physical Y (mm)', zaxis_title='Physical Z (mm)'),
                      height=650, margin=dict(l=0, r=0, b=0, t=100), legend=dict(itemsizing='constant'))
    return fig


def save_overview(report, output_path, edges=DEFAULT_EDGES):
    """Export representative Z-slice overlays, including the largest plaque slice."""
    import matplotlib.pyplot as plt

    _validate(report, edges)
    selection = report.mask == 2
    if not selection.any():
        selection = report.mask == 1
    active = np.flatnonzero(selection.any(axis=(1, 2)))
    if len(active):
        choices = [initial_slice(report.mask)] + active[np.linspace(0, len(active)-1, 3).astype(int)].tolist()
        indices = sorted(set(choices))
    else:
        indices = [report.mask.shape[0] // 2]
    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 5), squeeze=False)
    for ax, index in zip(axes[0], indices):
        ct, overlay, extent = slice_data(report, 0, index, edges)
        ax.imshow(ct, cmap='gray', vmin=-200, vmax=900, extent=extent, interpolation='nearest', aspect='equal')
        ax.imshow(overlay, extent=extent, interpolation='nearest', aspect='equal')
        ax.set_title(f'Z slice {index}')
        ax.set_xlabel('mm')
        ax.set_ylabel('mm')
    no_plaque = ' — no model-labeled plaque' if not (report.mask == 2).any() else ''
    fig.suptitle(f'{report.name}: vessel and plaque overlays{no_plaque}')
    fig.legend(handles=_legend(), loc='lower center', ncol=4 if len(indices) > 1 else 2, fontsize=8)
    fig.tight_layout(rect=(0, 0.20, 1, 0.92))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
