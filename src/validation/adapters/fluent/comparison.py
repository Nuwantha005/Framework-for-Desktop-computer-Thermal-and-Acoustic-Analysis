"""
BL comparison runner and metrics.

This module orchestrates the complete comparison pipeline between the
panel-method boundary layer solver and Fluent CFD results. It produces
comparison results suitable for visualization and error analysis.

Main classes:
- :class:`BLComparisonMetrics` — error metrics for one quantity
- :class:`BLComparisonResult` — complete comparison result
- :class:`BLComparisonRunner` — orchestrates the comparison pipeline
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from .ascii_reader import load_fluent_bl_data
from .bl_extractor import FluentBLExtractor
from .data_types import FluentBLResult
from .interpolator import (
    BLFieldInterpolator,
    InterpolatedBLField,
    create_interpolated_field,
)

if TYPE_CHECKING:
    from core.io import Case
    from solvers.boundary_layer.field import BLFieldData
    from solvers.boundary_layer.runner import BoundaryLayerCaseResult


@dataclass
class BLComparisonMetrics:
    """Error metrics for a single quantity comparison.

    All metrics are computed over valid (non-NaN) points only.

    Attributes:
        L2: L2 (Euclidean) norm of error.
        L_inf: L-infinity (maximum) norm of error.
        RMS: Root mean square error.
        MAE: Mean absolute error.
        relative_L2: L2 error normalized by reference L2 norm.
        n_points: Number of valid comparison points.
    """

    L2: float
    L_inf: float
    RMS: float
    MAE: float
    relative_L2: float
    n_points: int

    @classmethod
    def compute(
        cls,
        reference: NDArray[np.float64],
        test: NDArray[np.float64],
    ) -> "BLComparisonMetrics":
        """Compute metrics between reference and test arrays.

        Args:
            reference: Reference values (e.g., Fluent).
            test: Test values (e.g., BL solver).

        Returns:
            :class:`BLComparisonMetrics` instance.
        """
        # Flatten and filter NaN
        ref_flat = reference.ravel()
        test_flat = test.ravel()

        valid = ~(np.isnan(ref_flat) | np.isnan(test_flat))
        ref_valid = ref_flat[valid]
        test_valid = test_flat[valid]

        n = len(ref_valid)
        if n == 0:
            return cls(
                L2=np.nan,
                L_inf=np.nan,
                RMS=np.nan,
                MAE=np.nan,
                relative_L2=np.nan,
                n_points=0,
            )

        error = test_valid - ref_valid

        L2 = float(np.sqrt(np.sum(error**2)))
        L_inf = float(np.max(np.abs(error)))
        RMS = float(np.sqrt(np.mean(error**2)))
        MAE = float(np.mean(np.abs(error)))

        ref_L2 = float(np.sqrt(np.sum(ref_valid**2)))
        relative_L2 = L2 / ref_L2 if ref_L2 > 1e-12 else np.nan

        return cls(
            L2=L2,
            L_inf=L_inf,
            RMS=RMS,
            MAE=MAE,
            relative_L2=relative_L2,
            n_points=n,
        )


@dataclass
class BLComparisonResult:
    """Complete comparison result between BL solver and Fluent.

    Contains the original BL solver results, extracted Fluent BL data,
    interpolated Fluent fields on the BL solver grid, and error metrics.

    Attributes:
        bl_result: Panel-method BL solver result.
        fluent_result: Extracted Fluent BL quantities (Ue, δ, Cf, etc.).
        upper_fluent_field: Interpolated Fluent velocity on upper BL grid.
        lower_fluent_field: Interpolated Fluent velocity on lower BL grid.
        velocity_metrics: Velocity field error metrics per side and profile.
        wall_metrics: Wall quantity (Ue, δ, Cf) metrics per side.
        profile_name: Name of the BL profile used for comparison.
    """

    bl_result: "BoundaryLayerCaseResult"
    fluent_result: Optional[FluentBLResult]

    # Interpolated Fluent fields on BL solver grid
    upper_fluent_field: Optional[InterpolatedBLField] = None
    lower_fluent_field: Optional[InterpolatedBLField] = None

    # Error metrics
    # Structure: {side: {quantity: metrics}}
    velocity_metrics: Dict[str, Dict[str, BLComparisonMetrics]] = field(
        default_factory=dict
    )
    wall_metrics: Dict[str, Dict[str, BLComparisonMetrics]] = field(
        default_factory=dict
    )

    profile_name: str = ""

    @property
    def has_fluent_data(self) -> bool:
        """Whether Fluent comparison data is available."""
        return self.fluent_result is not None

    @property
    def sides(self) -> Dict[str, Optional[InterpolatedBLField]]:
        """Convenience mapping {"upper": field, "lower": field}."""
        return {
            "upper": self.upper_fluent_field,
            "lower": self.lower_fluent_field,
        }


class BLComparisonRunner:
    """Orchestrate BL comparison between panel method and Fluent.

    This class handles the complete comparison pipeline:
    1. Load Fluent ASCII exports
    2. Extract BL quantities from Fluent data
    3. Interpolate Fluent onto BL solver grid
    4. Compute error metrics
    5. Return comparison result for visualization

    Args:
        case: Panel-method case with mesh and fluid configuration.
        bl_result: Boundary layer solver result to compare.

    Example::

        from validation.adapters.fluent import BLComparisonRunner
        from solvers.boundary_layer.runner import BoundaryLayerRunner

        # Run BL solver
        runner = BoundaryLayerRunner(case, solver)
        bl_result = runner.run(reconstruct=True)

        # Run comparison
        comparison = BLComparisonRunner(case, bl_result)
        result = comparison.run(profile_name="Thwaites")

        # Visualize
        if result.has_fluent_data:
            plot_bl_fluent_comparison(
                bl_result.upper.fields["Thwaites"],
                result.upper_fluent_field,
            )
    """

    def __init__(
        self,
        case: "Case",
        bl_result: "BoundaryLayerCaseResult",
    ) -> None:
        self.case = case
        self.bl_result = bl_result

        # Extract fluid properties from case
        fluid = case.config.fluid
        self.rho = fluid.density
        freestream_vel = case.config.get_freestream_velocity()
        self.U_inf = float(np.linalg.norm(freestream_vel))
        # Fluent exports gauge pressure (relative to operating pressure),
        # so P0_inf should be the stagnation gauge pressure = 0.5 * rho * U_inf^2
        self.P0_inf = 0.5 * self.rho * self.U_inf**2

    def _normalize_profile_name(self, profile_name: str) -> str:
        """Find the actual profile name matching the given name (case-insensitive).
        
        The profile names in BL solver use capitalized forms like 'Thwaites',
        'Blasius', etc. This helper finds the matching key.
        """
        # Get available profile names from BL result
        available = self.bl_result.profile_names
        
        # Try exact match first
        if profile_name in available:
            return profile_name
        
        # Try case-insensitive match
        lower_name = profile_name.lower()
        for name in available:
            if name.lower() == lower_name:
                return name
        
        # No match found, return original (will fail later with clear error)
        return profile_name

    def run(
        self,
        profile_name: str = "thwaites",
    ) -> BLComparisonResult:
        """Execute the comparison pipeline.

        Args:
            profile_name: Name of BL profile to compare (must have
                reconstruction enabled in the BL solver run). Case-insensitive.

        Returns:
            :class:`BLComparisonResult` with comparison data and metrics.
            If Fluent data is unavailable, returns a result with
            ``has_fluent_data=False``.
        """
        # Normalize profile name to match actual BL solver keys
        profile_name = self._normalize_profile_name(profile_name)
        
        # Try to load Fluent data
        case_dir = Path(self.case.case_dir)
        field_data, wall_data = load_fluent_bl_data(case_dir)

        if field_data is None or wall_data is None:
            warnings.warn(
                "Fluent data unavailable — returning comparison result without "
                "Fluent fields. Visualization will show BL solver only.",
                stacklevel=2,
            )
            return BLComparisonResult(
                bl_result=self.bl_result,
                fluent_result=None,
                profile_name=profile_name,
            )

        # Extract BL quantities from Fluent
        extractor = FluentBLExtractor(
            bl_result=self.bl_result,
            field_data=field_data,
            wall_data=wall_data,
            rho=self.rho,
            U_inf=self.U_inf,
            P0_inf=self.P0_inf,
        )
        fluent_result = extractor.extract()

        # Interpolate Fluent onto BL solver grid for velocity comparison
        upper_fluent_field = self._interpolate_for_side(
            "upper", profile_name, field_data, fluent_result
        )
        lower_fluent_field = self._interpolate_for_side(
            "lower", profile_name, field_data, fluent_result
        )

        # Compute error metrics
        velocity_metrics = self._compute_velocity_metrics(
            profile_name, upper_fluent_field, lower_fluent_field
        )
        wall_metrics = self._compute_wall_metrics(fluent_result)

        return BLComparisonResult(
            bl_result=self.bl_result,
            fluent_result=fluent_result,
            upper_fluent_field=upper_fluent_field,
            lower_fluent_field=lower_fluent_field,
            velocity_metrics=velocity_metrics,
            wall_metrics=wall_metrics,
            profile_name=profile_name,
        )

    def _interpolate_for_side(
        self,
        side: str,
        profile_name: str,
        field_data,
        fluent_result: FluentBLResult,
    ) -> Optional[InterpolatedBLField]:
        """Interpolate Fluent data onto BL solver grid for one side."""
        bl_path = self.bl_result.sides[side]
        fluent_path = fluent_result.sides[side]

        # Check if reconstruction is available
        if profile_name not in bl_path.fields:
            warnings.warn(
                f"BL field reconstruction not available for profile "
                f"'{profile_name}' on {side} side. Run BL solver with "
                f"reconstruct=True.",
                stacklevel=3,
            )
            return None

        bl_field = bl_path.fields[profile_name]

        # Create interpolator
        interpolator = BLFieldInterpolator(
            bl_path=bl_path,
            surface_x=self.bl_result.surface_x,
            surface_y=self.bl_result.surface_y,
            panel_normals=self.case.mesh.normals,
        )

        # Transform BL grid to global coordinates
        X, Y = interpolator.transform_to_global(bl_field)

        # Interpolate Fluent velocity
        fluent_u = interpolator.interpolate_fluent_velocity(X, Y, field_data)

        # Interpolate edge velocity from Fluent wall data
        fluent_Ue = np.interp(
            bl_field.s,
            fluent_path.s,
            fluent_path.Ue,
            left=np.nan,
            right=np.nan,
        )

        return create_interpolated_field(bl_field, fluent_u, fluent_Ue)

    def _compute_velocity_metrics(
        self,
        profile_name: str,
        upper_fluent_field: Optional[InterpolatedBLField],
        lower_fluent_field: Optional[InterpolatedBLField],
    ) -> Dict[str, Dict[str, BLComparisonMetrics]]:
        """Compute velocity field error metrics."""
        metrics: Dict[str, Dict[str, BLComparisonMetrics]] = {}

        for side, fluent_field in [
            ("upper", upper_fluent_field),
            ("lower", lower_fluent_field),
        ]:
            if fluent_field is None:
                continue

            bl_path = self.bl_result.sides[side]
            if profile_name not in bl_path.fields:
                continue

            bl_field = bl_path.fields[profile_name]

            metrics[side] = {
                "u": BLComparisonMetrics.compute(fluent_field.u, bl_field.u),
                "Ue": BLComparisonMetrics.compute(
                    fluent_field.Ue, bl_field.Ue
                ),
            }

        return metrics

    def _compute_wall_metrics(
        self,
        fluent_result: FluentBLResult,
    ) -> Dict[str, Dict[str, BLComparisonMetrics]]:
        """Compute wall quantity (Ue, δ, Cf) error metrics."""
        metrics: Dict[str, Dict[str, BLComparisonMetrics]] = {}

        for side in ["upper", "lower"]:
            bl_path = self.bl_result.sides[side]
            fluent_path = fluent_result.sides[side]

            # Find common arc-length range
            s_min = max(bl_path.s.min(), fluent_path.s.min())
            s_max = min(bl_path.s.max(), fluent_path.s.max())

            if s_max <= s_min:
                continue

            # Interpolate to common stations
            s_common = np.linspace(s_min, s_max, 50)

            # Get first available profile result
            if not bl_path.results:
                continue
            first_profile = list(bl_path.results.keys())[0]
            bl_res = bl_path.results[first_profile]

            # Interpolate BL solver quantities
            bl_Ue = np.interp(s_common, bl_path.s, bl_path.Ue)
            bl_cf_local = np.interp(s_common, bl_res.s, bl_res.cf)
            # Convert BL solver cf (based on local Ue) to freestream-based Cf
            # cf_local = 2 * τ_w / (ρ * Ue²)
            # Cf_freestream = τ_w / (0.5 * ρ * U_inf²)
            # => Cf_freestream = cf_local * (Ue / U_inf)²
            bl_Cf = bl_cf_local * (bl_Ue / self.U_inf) ** 2

            # Interpolate Fluent quantities
            fluent_Ue = np.interp(s_common, fluent_path.s, fluent_path.Ue)
            fluent_cf = np.interp(s_common, fluent_path.s, fluent_path.Cf)

            metrics[side] = {
                "Ue": BLComparisonMetrics.compute(fluent_Ue, bl_Ue),
                "Cf": BLComparisonMetrics.compute(fluent_cf, bl_Cf),
            }

            # Add delta comparison if available
            if first_profile in bl_path.fields:
                bl_delta = bl_path.fields[first_profile].delta
                bl_delta_interp = np.interp(
                    s_common,
                    bl_path.fields[first_profile].s,
                    bl_delta,
                )
                fluent_delta_interp = np.interp(
                    s_common, fluent_path.s, fluent_path.delta
                )
                metrics[side]["delta"] = BLComparisonMetrics.compute(
                    fluent_delta_interp, bl_delta_interp
                )

        return metrics
