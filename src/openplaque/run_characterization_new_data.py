"""End-to-end optimized boundary refinement plus plaque characterization for new data."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from .run_new_data import (
    load_best_boundary_parameters,
    auto_detect_or_fallback_series,
    refine_reports_with_parameters,
    core_reports_with_parameters,
    tpv_summary_rows,
    save_tpv_summary_csv,
    save_refined_masks,
    save_overlays,
    write_new_data_html_report,
)
from .plaque_characterization import (
    DEFAULT_THRESHOLDS,
    summarize_all_vessels,
    save_characterization_csv,
    save_lesion_csv,
    save_class_maps_nifti,
    save_type_overlays,
    write_characterization_html_report,
)


def process_new_study_with_characterization(
    study_zip: str | Path,
    best_parameters_json: str | Path,
    output_dir: str | Path,
    fallback_series: Optional[Mapping[str, int]] = None,
    thresholds: Optional[Mapping[str, Mapping[str, Optional[float]]]] = None,
    save_masks: bool = True,
    save_png_overlays: bool = True,
    save_class_maps: bool = True,
) -> dict[str, Any]:
    """Process a new CCTA study and quantify plaque composition by HU class."""
    from openplaque.study import OpenPlaqueStudy
    from openplaque.segmentation import segment_vessel

    params = load_best_boundary_parameters(best_parameters_json)
    thresholds = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    study = OpenPlaqueStudy(str(study_zip))
    series_map = auto_detect_or_fallback_series(study, fallback_series=fallback_series)

    reports = []
    for artery in ["LAD", "RCA", "LCX"]:
        image, volume, _ = study.load_series(series_map[artery])
        report = segment_vessel(image, volume, artery)
        reports.append(report)

    refinements = refine_reports_with_parameters(reports, params)
    core_results = core_reports_with_parameters(reports, params)

    tpv_rows = tpv_summary_rows(reports, refinements, core_results=core_results)
    tpv_csv = save_tpv_summary_csv(tpv_rows, output_dir / "optimized_tpv_summary.csv")
    mask_paths = save_refined_masks(reports, refinements, output_dir / "masks") if save_masks else []
    boundary_overlay_paths = save_overlays(reports, refinements, output_dir / "overlays") if save_png_overlays else []
    boundary_html = write_new_data_html_report(output_dir / "optimized_boundary_tpv_report.html", tpv_rows, params, overlay_paths=boundary_overlay_paths)

    characterization_summaries, lesion_summaries, class_maps = summarize_all_vessels(reports, refinements, thresholds=thresholds, connectivity=int(params.get("connectivity", 26)))
    composition_csv = save_characterization_csv(characterization_summaries, output_dir / "plaque_type_composition_summary.csv")
    lesion_csv = save_lesion_csv(lesion_summaries, output_dir / "plaque_type_lesion_summary.csv")
    class_map_paths = save_class_maps_nifti(reports, class_maps, output_dir / "masks") if save_class_maps else []
    type_overlay_paths = save_type_overlays(reports, class_maps, output_dir / "overlays") if save_png_overlays else []
    characterization_html = write_characterization_html_report(
        output_dir / "plaque_characterization_report.html",
        characterization_summaries,
        lesion_summaries,
        thresholds=thresholds,
        overlay_paths=type_overlay_paths,
    )

    return {
        "params": params,
        "thresholds": thresholds,
        "series_map": series_map,
        "reports": reports,
        "refinements": refinements,
        "core_results": core_results,
        "tpv_rows": tpv_rows,
        "characterization_summaries": characterization_summaries,
        "lesion_summaries": lesion_summaries,
        "class_maps": class_maps,
        "tpv_csv": tpv_csv,
        "composition_csv": composition_csv,
        "lesion_csv": lesion_csv,
        "mask_paths": mask_paths,
        "class_map_paths": class_map_paths,
        "boundary_overlay_paths": boundary_overlay_paths,
        "type_overlay_paths": type_overlay_paths,
        "boundary_html": boundary_html,
        "characterization_html": characterization_html,
    }
