"""
Boundary layer solvers — Von Kármán momentum integral method.

Provides :class:`BoundaryLayerSolver`, pluggable velocity profiles, and
transition prediction utilities.

Quick start::

    from solvers.boundary_layer import BoundaryLayerSolver
    from solvers.boundary_layer.profiles import BlasiusProfile

    bl = BoundaryLayerSolver(
        edge_velocity=Ue,
        arc_length=s,
        nu=1.5e-5,
        profile=BlasiusProfile(),
    )
    result = bl.solve()
"""

from .base import BoundaryLayerSolver, BoundaryLayerResult
from .field import BLFieldData, reconstruct_bl_field
from .transition import (
    TransitionResult,
    michel_criterion,
    en_criterion,
)
from .profiles import (
    VelocityProfile,
    BlasiusProfile,
    PohlhausenProfile,
    FalknerSkanProfile,
    PowerLawProfile,
    ThwaitesProfile,
)

__all__ = [
    # Solver & result
    "BoundaryLayerSolver",
    "BoundaryLayerResult",
    # Field reconstruction
    "BLFieldData",
    "reconstruct_bl_field",
    # Profiles
    "VelocityProfile",
    "BlasiusProfile",
    "PohlhausenProfile",
    "FalknerSkanProfile",
    "PowerLawProfile",
    "ThwaitesProfile",
    # Transition
    "TransitionResult",
    "michel_criterion",
    "en_criterion",
]
