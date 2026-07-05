"""TPV uncertainty summaries for OpenPlaque."""

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Mapping


@dataclass
class VesselTPVUncertainty:
    vessel: str
    core_tpv_mm3: float
    refined_tpv_mm3: float
    raw_tpv_mm3: float
    removed_boundary_mm3: float

    @property
    def interval_low_mm3(self) -> float:
        return self.core_tpv_mm3

    @property
    def interval_high_mm3(self) -> float:
        return self.raw_tpv_mm3

    @property
    def interval_mid_mm3(self) -> float:
        return self.refined_tpv_mm3

    @property
    def uncertainty_width_mm3(self) -> float:
        return self.interval_high_mm3 - self.interval_low_mm3

    def to_dict(self):
        d = asdict(self)
        d.update({
            "interval_low_mm3": self.interval_low_mm3,
            "interval_mid_mm3": self.interval_mid_mm3,
            "interval_high_mm3": self.interval_high_mm3,
            "uncertainty_width_mm3": self.uncertainty_width_mm3,
        })
        return d


@dataclass
class TPVUncertaintySummary:
    vessels: Dict[str, VesselTPVUncertainty]

    @property
    def total_core_tpv_mm3(self) -> float:
        return sum(v.core_tpv_mm3 for v in self.vessels.values())

    @property
    def total_refined_tpv_mm3(self) -> float:
        return sum(v.refined_tpv_mm3 for v in self.vessels.values())

    @property
    def total_raw_tpv_mm3(self) -> float:
        return sum(v.raw_tpv_mm3 for v in self.vessels.values())

    @property
    def total_removed_boundary_mm3(self) -> float:
        return sum(v.removed_boundary_mm3 for v in self.vessels.values())

    @property
    def total_uncertainty_width_mm3(self) -> float:
        return self.total_raw_tpv_mm3 - self.total_core_tpv_mm3

    def total_row(self) -> dict:
        return {
            "vessel": "TOTAL",
            "core_tpv_mm3": self.total_core_tpv_mm3,
            "refined_tpv_mm3": self.total_refined_tpv_mm3,
            "raw_tpv_mm3": self.total_raw_tpv_mm3,
            "removed_boundary_mm3": self.total_removed_boundary_mm3,
            "interval_low_mm3": self.total_core_tpv_mm3,
            "interval_mid_mm3": self.total_refined_tpv_mm3,
            "interval_high_mm3": self.total_raw_tpv_mm3,
            "uncertainty_width_mm3": self.total_uncertainty_width_mm3,
        }

    def rows(self):
        return [v.to_dict() for v in self.vessels.values()] + [self.total_row()]


def make_tpv_uncertainty_summary(raw_reports: Iterable, refined_results: Mapping[str, object], core_results: Mapping[str, object]) -> TPVUncertaintySummary:
    vessels = {}
    for report in raw_reports:
        name = report.name
        vessels[name] = VesselTPVUncertainty(
            vessel=name,
            core_tpv_mm3=float(core_results[name].refined_tpv_mm3),
            refined_tpv_mm3=float(refined_results[name].refined_tpv_mm3),
            raw_tpv_mm3=float(report.tpv_mm3),
            removed_boundary_mm3=float(refined_results[name].removed_volume_mm3),
        )
    return TPVUncertaintySummary(vessels=vessels)
