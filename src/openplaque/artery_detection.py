"""
Automatic coronary artery curved-reformat series detection for OpenPlaque.

This module replaces hard-coded series numbers with metadata-based heuristics.
It accepts an OpenPlaqueStudy-like object and can optionally inspect a DICOM ZIP
when metadata is not exposed by the study object.

Research use only. Not for clinical decision-making.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import re
import tempfile
import zipfile


ARTERY_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "LAD": ("lad", "left anterior descending"),
    "RCA": ("rca", "right coronary"),
    "LCX": ("lcx", "cx", "circumflex", "left circumflex"),
}

CURVED_KEYWORDS = (
    "curved", "cpr", "curved planar", "reformat", "reformation", "mpr",
    "vessel", "coronary", "straightened", "stretched",
)


@dataclass
class ArterySeriesCandidate:
    artery: str
    series_number: int
    score: float
    description: str = ""
    reason: str = ""


def _normalise_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _get_attr(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _series_number_from_record(record: Any) -> Optional[int]:
    value = _get_attr(record, ("series_number", "SeriesNumber", "number", "id", "series"))
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        digits = re.findall(r"\d+", str(value))
        return int(digits[-1]) if digits else None


def _description_from_record(record: Any) -> str:
    fields = [
        "description", "SeriesDescription", "series_description", "name",
        "ProtocolName", "protocol_name", "ImageType", "image_type",
        "BodyPartExamined", "body_part_examined",
    ]
    parts = []
    for field in fields:
        value = _get_attr(record, (field,))
        if value:
            if isinstance(value, (list, tuple)):
                parts.append(" ".join(map(str, value)))
            else:
                parts.append(str(value))
    return " | ".join(parts)


def _records_from_study_attributes(study: Any) -> List[Any]:
    for attr in ("series", "series_info", "series_metadata", "dicom_series", "series_records"):
        value = getattr(study, attr, None)
        if value:
            if isinstance(value, Mapping):
                records = []
                for key, record in value.items():
                    if isinstance(record, Mapping):
                        merged = dict(record)
                        merged.setdefault("series_number", key)
                        records.append(merged)
                    else:
                        records.append(record)
                return records
            return list(value)

    for method in ("list_series", "get_series", "available_series", "series_summary"):
        fn = getattr(study, method, None)
        if callable(fn):
            try:
                value = fn()
                if value:
                    if isinstance(value, Mapping):
                        out = []
                        for key, record in value.items():
                            if isinstance(record, Mapping):
                                merged = dict(record)
                                merged.setdefault("series_number", key)
                                out.append(merged)
                            else:
                                out.append(record)
                        return out
                    return list(value)
            except Exception:
                pass
    return []


def _dicom_zip_path_from_study(study: Any) -> Optional[Path]:
    for attr in ("zip_path", "study_zip", "dicom_zip", "path", "source", "input_path"):
        value = getattr(study, attr, None)
        if value and str(value).lower().endswith(".zip"):
            p = Path(str(value))
            if p.exists():
                return p
    return None


def _records_from_dicom_zip(zip_path: Path, max_files: int = 250) -> List[dict]:
    """Read a small sample of DICOM metadata from a ZIP, grouped by SeriesNumber."""
    try:
        import pydicom
    except Exception:
        return []

    records_by_series: Dict[int, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            # Prefer likely DICOM files but keep this permissive.
            names = names[:max_files]
            for name in names:
                try:
                    target = tmpdir / Path(name).name
                    target.write_bytes(zf.read(name))
                    ds = pydicom.dcmread(str(target), stop_before_pixels=True, force=True)
                    sn = getattr(ds, "SeriesNumber", None)
                    if sn is None:
                        continue
                    sn = int(sn)
                    if sn not in records_by_series:
                        records_by_series[sn] = {
                            "series_number": sn,
                            "SeriesDescription": getattr(ds, "SeriesDescription", ""),
                            "ProtocolName": getattr(ds, "ProtocolName", ""),
                            "ImageType": " ".join(map(str, getattr(ds, "ImageType", []))),
                            "BodyPartExamined": getattr(ds, "BodyPartExamined", ""),
                        }
                except Exception:
                    continue
    return list(records_by_series.values())


def _records_from_study(study: Any) -> List[Any]:
    records = _records_from_study_attributes(study)
    if records:
        return records
    zip_path = _dicom_zip_path_from_study(study)
    if zip_path:
        return _records_from_dicom_zip(zip_path)
    return []


def score_series_for_artery(record: Any, artery: str) -> ArterySeriesCandidate:
    artery = artery.upper()
    number = _series_number_from_record(record)
    description = _description_from_record(record)
    text = _normalise_text(description)

    score = 0.0
    reasons: List[str] = []

    for alias in ARTERY_ALIASES[artery]:
        alias_text = _normalise_text(alias)
        if re.search(rf"(^|[^a-z0-9]){re.escape(alias_text)}([^a-z0-9]|$)", text):
            score += 100
            reasons.append(f"matched artery alias '{alias}'")
            break

    curved_hits = [kw for kw in CURVED_KEYWORDS if kw in text]
    if curved_hits:
        score += min(30, 10 * len(curved_hits))
        reasons.append("curved/coronary keyword match")

    if number and number >= 1000:
        score += 5
        reasons.append("high derived/reformat series number")

    return ArterySeriesCandidate(
        artery=artery,
        series_number=number if number is not None else -1,
        score=score,
        description=description,
        reason="; ".join(reasons) or "no strong metadata match",
    )


def detect_artery_series(
    study: Any,
    required: Sequence[str] = ("LAD", "RCA", "LCX"),
    fallback_series: Optional[Mapping[str, int]] = None,
    min_score: float = 100.0,
    return_candidates: bool = False,
):
    """
    Detect likely LAD/RCA/LCX curved-reformat series numbers.

    If metadata is unavailable or ambiguous, `fallback_series` is used. For the
    UCLA example, pass {"RCA": 1035, "LCX": 1039, "LAD": 1043}.
    """
    records = _records_from_study(study)
    fallback = {k.upper(): int(v) for k, v in (fallback_series or {}).items()}
    detected: Dict[str, int] = {}
    all_candidates: Dict[str, List[ArterySeriesCandidate]] = {}

    for artery in [a.upper() for a in required]:
        candidates = [score_series_for_artery(r, artery) for r in records]
        candidates = [c for c in candidates if c.series_number >= 0]
        candidates.sort(key=lambda c: c.score, reverse=True)
        all_candidates[artery] = candidates[:10]

        if candidates and candidates[0].score >= min_score:
            detected[artery] = candidates[0].series_number
        elif artery in fallback:
            detected[artery] = fallback[artery]
        else:
            top = candidates[0] if candidates else None
            detail = f" Top candidate was {top.series_number} score={top.score:.1f}: {top.description}" if top else ""
            raise ValueError(
                f"Could not confidently detect {artery} curved-reformat series."
                f" Provide fallback_series. {detail}"
            )

    if return_candidates:
        return detected, all_candidates
    return detected


def load_detected_arteries(
    study: Any,
    fallback_series: Optional[Mapping[str, int]] = None,
    required: Sequence[str] = ("LAD", "RCA", "LCX"),
) -> Dict[str, Tuple[Any, Any, Any]]:
    series_map = detect_artery_series(study, required=required, fallback_series=fallback_series)
    return {artery: study.load_series(series_number) for artery, series_number in series_map.items()}
