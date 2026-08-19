"""Data structures shared by the temporal analyses."""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass
class TemporalContext:
    months: list[str]
    packet_rows_by_month: dict[str, np.ndarray]
    sensor_total_by_month: dict[str, dict[str, int]]
    total_packets_by_month: dict[str, int]
    pattern_lookup_by_month: dict[
        str, dict[str, list[tuple[frozenset[str], int]]]
    ]


@dataclass
class CombinedWindow:
    months: list[str]
    rows: np.ndarray
    packet_matrix: np.ndarray
    packet_sensor: np.ndarray
    sensor_total: dict[str, int]
    sensors_active: list[str]
    pattern_lookup: dict[str, list[tuple[frozenset[str], int]]]
    total_packets: int


@dataclass
class FitResult:
    status: str
    n_links: int | None
    selected_links: list[str]

    @classmethod
    def from_mapping(cls, result: Mapping[str, Any]) -> "FitResult":
        links = result.get("selected_links") or []
        return cls(
            status=str(result["status"]),
            n_links=result.get("n_links"),
            selected_links=sorted(links),
        )


@dataclass
class SubsetEvaluation:
    packet_coverage_pct: float
    worst_sensor_coverage_pct: float
    median_sensor_coverage_pct: float
    n_active_sensors: int

    def to_record(self) -> dict[str, float | int]:
        return {
            "packet_coverage_pct": self.packet_coverage_pct,
            "worst_sensor_coverage_pct": self.worst_sensor_coverage_pct,
            "median_sensor_coverage_pct": self.median_sensor_coverage_pct,
            "n_active_sensors": self.n_active_sensors,
        }
