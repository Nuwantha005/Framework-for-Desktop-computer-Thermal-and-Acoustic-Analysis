"""IO utilities: geometry readers, case loader, exporters."""

from .geometry_io import GeometryReader, generate_rectangle, generate_circle
from .case_loader import CaseLoader, create_example_case

__all__ = [
    "GeometryReader",
    "generate_rectangle",
    "generate_circle",
    "CaseLoader",
    "create_example_case",
]
