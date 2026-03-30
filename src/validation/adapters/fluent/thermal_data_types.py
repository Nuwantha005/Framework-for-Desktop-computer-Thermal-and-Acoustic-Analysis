"""Data types for Fluent thermal boundary-layer comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from .comparison import BLComparisonMetrics


@dataclass
class FluentThermalWallData:
    """Raw Fluent thermal wall export data."""

    node_id: NDArray[np.int32]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    temperature: NDArray[np.float64]
    heat_transfer_coeff: NDArray[np.float64]


@dataclass
class FluentThermalFieldData:
    """Raw Fluent thermal field export data."""

    node_id: NDArray[np.int32]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    temperature: NDArray[np.float64]

    @property
    def points(self) -> NDArray[np.float64]:
        """Node coordinates as (N, 2) array."""
        return np.column_stack([self.x, self.y])


@dataclass
class FluentThermalPathResult:
    """Thermal wall quantities on one BL path."""

    side: str
    s: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    wall_temperature: NDArray[np.float64]
    heat_transfer_coeff: NDArray[np.float64]


@dataclass
class FluentThermalResult:
    """Two-sided Fluent thermal wall quantities."""

    upper: FluentThermalPathResult
    lower: FluentThermalPathResult

    @property
    def sides(self) -> Dict[str, FluentThermalPathResult]:
        return {"upper": self.upper, "lower": self.lower}


@dataclass
class InterpolatedThermalField:
    """Interpolated Fluent temperature field on thermal solver grid."""

    s: NDArray[np.float64]
    y: NDArray[np.float64]
    T: NDArray[np.float64]
    delta: NDArray[np.float64]
    T_inf: float
    source: str = "fluent"


@dataclass
class ThermalComparisonResult:
    """Complete thermal BL comparison result for visualization."""

    bl_result: object
    upper_thermal_result: object
    lower_thermal_result: object
    fluent_wall_result: Optional[FluentThermalResult]
    upper_panel_indices: List[int] = field(default_factory=list)
    lower_panel_indices: List[int] = field(default_factory=list)
    upper_fluent_field: Optional[InterpolatedThermalField] = None
    lower_fluent_field: Optional[InterpolatedThermalField] = None
    wall_metrics: Dict[str, Dict[str, BLComparisonMetrics]] = field(default_factory=dict)
    field_metrics: Dict[str, Dict[str, BLComparisonMetrics]] = field(default_factory=dict)

    @property
    def has_fluent_data(self) -> bool:
        return self.fluent_wall_result is not None

    @property
    def sides(self) -> Dict[str, Optional[InterpolatedThermalField]]:
        return {"upper": self.upper_fluent_field, "lower": self.lower_fluent_field}
