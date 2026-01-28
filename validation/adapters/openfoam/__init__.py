"""
OpenFOAM adapter for validation pipeline.

Provides tools for:
- Converting 2D panel geometry to 3D STL
- Generating OpenFOAM case structures (template-based with foamlib)
- Running OpenFOAM solvers
- Reading OpenFOAM results (via foamlib)
"""

from .case_generator import OpenFOAMCaseGenerator, MeshSettings  # Old generator (backward compat)
from .foamlib_generator import FoamlibCaseGenerator  # NEW: foamlib-based generator (shares MeshSettings)
from .runner import OpenFOAMRunner
from .geometry_converter import GeometryConverter
from .simple_generator import SimpleOpenFOAMGenerator
from .surface_extractor import OpenFOAMSurfaceExtractor

# Re-export foamlib classes for convenience
from foamlib import FoamCase, FoamFieldFile

__all__ = [
    # Generators
    'OpenFOAMCaseGenerator',  # Old (kept for backward compatibility)
    'FoamlibCaseGenerator',    # NEW: Recommended for new code
    'MeshSettings',
    'SimpleOpenFOAMGenerator',
    # Runner & utilities
    'OpenFOAMRunner',
    'GeometryConverter',
    'OpenFOAMSurfaceExtractor',
    # foamlib re-exports
    'FoamCase',
    'FoamFieldFile',
]
