"""Data containers for actuator disk coupling."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from core.config.schemas import ActuatorDiskConfig
from core.geometry.mesh3d import Mesh3D

from .fan_curve import FanCurve


@dataclass
class ADMIterationRecord:
    """Convergence record for one actuator disk at one iteration."""

    iteration: int
    disk_name: str
    flow_rate: float
    pressure_rise: float
    pressure_rise_curve: float
    pressure_residual: float


@dataclass
class ActuatorDiskRuntime:
    """Runtime representation of one configured actuator disk."""

    config: ActuatorDiskConfig
    mesh: Mesh3D
    curve: FanCurve
    normal: NDArray[np.float64]
    pressure_rise: float
    doublet_strength: NDArray[np.float64]
    reference_velocity: float
    normal_velocity: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    flow_rate: float = 0.0
    sample_offset: float = 0.0


@dataclass
class ActuatorDiskResult:
    """Final actuator disk operating point."""

    name: str
    flow_rate: float
    pressure_rise: float
    doublet_strength: NDArray[np.float64]
    normal_velocity: NDArray[np.float64]
    converged: bool
    iterations: int
    warning: str | None = None
