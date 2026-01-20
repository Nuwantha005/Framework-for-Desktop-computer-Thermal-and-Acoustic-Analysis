"""
Post-processing module.

Provides a modular pipeline for computing derived fields (pressure, temperature, etc.)
from primary solver outputs (velocity).

Key classes:
- FieldData: Container for all computed fields
- FluidState: Fluid properties (density, viscosity, etc.)
- PostProcessor: Base class for field computations
- PressureField: Computes pressure from velocity using Bernoulli
"""

from .fields import FieldData, ScalarField, VectorField
from .fluid import FluidState, ReferenceCondition
from .pipeline import PostProcessor, ProcessorPipeline
from .pressure import PressureProcessor
from .velocity_potential import VelocityPotentialProcessor

__all__ = [
    # Field containers
    "FieldData",
    "ScalarField", 
    "VectorField",
    # Fluid properties
    "FluidState",
    "ReferenceCondition",
    # Pipeline
    "PostProcessor",
    "ProcessorPipeline",
    # Processors
    "PressureProcessor",
    "VelocityPotentialProcessor",
]
