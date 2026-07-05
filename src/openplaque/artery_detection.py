"""
Automatic coronary artery curved-reformat series detection for OpenPlaque.

This module replaces hard-coded UCLA series numbers such as 1035/1039/1043 with
heuristics that inspect available DICOM series metadata and choose the most
likely RCA, LAD, and LCX/CX curved reformats.

Research use only. Not for clinical decision-making.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import re


ARTERY_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "LAD": ("lad", "left anterior descending"),
    "RCA": ("rca", "right coronary"),
    "LCX": ("lcx", "cx", "circumflex", "left circumflex"),
}

CURVED_KEYWORDS = (
    "curved", "cpr", "curved planar", "reformat", "reformation", "mpr",
    "vessel", "coronary",
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
    value = _get_attr(record, ("series_number", "SeriesNumber", "number", "id"))
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
    ]
    parts = []
    for field in fields:
        value = _get_attr(record, (field,))
        if value:
            parts.append(str(value))
    return " | ".join(parts)


def _records_from_study(study: Any) -> List[Any]:
    """Try common OpenPlaque/pydicom/SimpleITK study summary layouts."""
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
                        return [dict(v, series_number=k) if isinstance(v, Mapping) else v for k, v in value.items()]
                    return list(value)
            except Exception:
                pass

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

    # Prefer high-numbered derived/reformat series only after artery keyword match.
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
) -> Dict[str, int]:
    """
    Detect likely LAD/RCA/LCX curved-reformat series numbers.

    Parameters
    ----------
    study:
        An OpenPlaqueStudy-like object. The function looks for common metadata
        fields/methods. If metadata is unavailable, provide fallback_series.
    required:
        Arteries to detect.
    fallback_series:
        Optional mapping, e.g. {"RCA": 1035, "LCX": 1039, "LAD": 1043}.
        Used only when metadata cannot confidently identify an artery.
    min_score:
        Minimum metadata score required before accepting a candidate.

    Returns
    -------
    dict
        Mapping such as {"LAD": 1043, "RCA": 1035, "LCX": 1039}.
    """
    records = _records_from_study(study)
    fallback = {k.upper(): int(v) for k, v in (fallback_series or {}).items()}
    detected: Dict[str, int] = {}

    for artery in [a.upper() for a in required]:
        candidates = [score_series_for_artery(r, artery) for r in records]
        candidates = [c for c in candidates if c.series_number >= 0]
        candidates.sort(key=lambda c: c.score, reverse=True)

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

    return detected


def load_detected_arteries(
    study: Any,
    fallback_series: Optional[Mapping[str, int]] = None,
    required: Sequence[str] = ("LAD", "RCA", "LCX"),
) -> Dict[str, Tuple[Any, Any, Any]]:
    """
    Detect and load artery series using study.load_series(series_number).
    """
    series_map = detect_artery_series(study, required=required, fallback_series=fallback_series)
    loaded = {}
    for artery, series_number in series_map.items():
        loaded[artery] = study.load_series(series_number)
    return loaded
