"""
Validation module for comparing panel method results with external CFD.

This module provides tools for:
- Converting panel method geometry to OpenFOAM format
- Generating OpenFOAM case structures from YAML case files
- Running OpenFOAM solvers (potentialFoam, simpleFoam, etc.)
- Reading results via foamlib

Usage:
    from validation import OpenFOAMRunner, OpenFOAMCaseGenerator

    # Generate and run OpenFOAM case
    generator = OpenFOAMCaseGenerator(case)
    of_case_dir = generator.generate()
    
    runner = OpenFOAMRunner(of_case_dir)
    runner.run_all()
    
    # Read results via foamlib
    case = runner.foam_case
    U = case[-1]["U"].internal_field  # Velocity at latest time
    p = case[-1]["p"].internal_field  # Pressure at latest time
"""

from .adapters.openfoam.case_generator import OpenFOAMCaseGenerator
from .adapters.openfoam.runner import OpenFOAMRunner
from .adapters.openfoam.geometry_converter import GeometryConverter
from .adapters.openfoam.simple_generator import SimpleOpenFOAMGenerator

# Re-export foamlib classes for convenience
from foamlib import FoamCase, FoamFieldFile

__all__ = [
    'OpenFOAMCaseGenerator',
    'OpenFOAMRunner', 
    'GeometryConverter',
    'SimpleOpenFOAMGenerator',
    'FoamCase',
    'FoamFieldFile',
]
