"""
OpenFOAM adapter for validation pipeline.

Provides tools for:
- Converting 2D panel geometry to 3D STL
- Generating OpenFOAM case structures
- Running OpenFOAM solvers
- Reading OpenFOAM results (via foamlib)

Note: Result reading is done via foamlib.FoamCase which is exposed
through OpenFOAMRunner.foam_case property.
"""

from .case_generator import OpenFOAMCaseGenerator
from .runner import OpenFOAMRunner
from .geometry_converter import GeometryConverter
from .simple_generator import SimpleOpenFOAMGenerator
from .surface_extractor import OpenFOAMSurfaceExtractor

# Re-export foamlib classes for convenience
from foamlib import FoamCase, FoamFieldFile

__all__ = [
    'OpenFOAMCaseGenerator',
    'OpenFOAMRunner',
    'GeometryConverter',
    'SimpleOpenFOAMGenerator',
    'OpenFOAMSurfaceExtractor',
    'FoamCase',
    'FoamFieldFile',
]
