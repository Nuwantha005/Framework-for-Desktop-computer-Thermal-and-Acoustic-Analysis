"""
Fluent adapter for boundary layer comparison.

Public API for loading and processing Fluent export data for comparison
with the panel-method boundary layer solver.

Data Loading
------------
- :func:`load_fluent_bl_data` — load field and wall data from case directory
- :func:`read_field_data` — parse Fluent filed_data ASCII export
- :func:`read_wall_data` — parse Fluent wall_data ASCII export

BL Extraction
-------------
- :class:`FluentBLExtractor` — extract BL quantities from Fluent data

Comparison
----------
- :class:`BLComparisonRunner` — orchestrate comparison pipeline
- :class:`BLComparisonResult` — complete comparison result
- :class:`BLComparisonMetrics` — error metrics for one quantity

Data Types
----------
- :class:`FluentFieldData` — raw field data container
- :class:`FluentWallData` — raw wall data container
- :class:`FluentBLResult` — extracted BL quantities
- :class:`InterpolatedBLField` — Fluent data on BL solver grid
"""

from .data_types import (
    FluentFieldData,
    FluentWallData,
    FluentBLPathResult,
    FluentBLResult,
)
from .ascii_reader import (
    read_field_data,
    read_wall_data,
    find_fluent_export_dir,
    load_fluent_bl_data,
)
from .bl_extractor import FluentBLExtractor
from .interpolator import (
    BLFieldInterpolator,
    InterpolatedBLField,
    create_interpolated_field,
)
from .comparison import (
    BLComparisonMetrics,
    BLComparisonResult,
    BLComparisonRunner,
)

__all__ = [
    # Data types
    "FluentFieldData",
    "FluentWallData",
    "FluentBLPathResult",
    "FluentBLResult",
    "InterpolatedBLField",
    # ASCII readers
    "read_field_data",
    "read_wall_data",
    "find_fluent_export_dir",
    "load_fluent_bl_data",
    # BL extraction
    "FluentBLExtractor",
    # Interpolation
    "BLFieldInterpolator",
    "create_interpolated_field",
    # Comparison
    "BLComparisonMetrics",
    "BLComparisonResult",
    "BLComparisonRunner",
]
