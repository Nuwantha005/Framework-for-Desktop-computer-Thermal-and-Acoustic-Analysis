"""
Adapters for different CFD solvers.
"""

from . import openfoam
from . import fluent

__all__ = ['openfoam', 'fluent']
