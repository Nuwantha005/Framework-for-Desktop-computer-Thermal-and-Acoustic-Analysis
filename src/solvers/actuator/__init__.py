"""Actuator disk model utilities and coupled 3D solver."""

from .coupling import ActuatorDiskCoupledSolver3D
from .disk_mesh import generate_actuator_disk_mesh, generate_rectangular_boundary_mesh
from .fan_curve import FanCurve
from .models import ActuatorDiskResult, ActuatorDiskRuntime, ADMIterationRecord

__all__ = [
    "ActuatorDiskCoupledSolver3D",
    "ActuatorDiskResult",
    "ActuatorDiskRuntime",
    "ADMIterationRecord",
    "FanCurve",
    "generate_actuator_disk_mesh",
    "generate_rectangular_boundary_mesh",
]
