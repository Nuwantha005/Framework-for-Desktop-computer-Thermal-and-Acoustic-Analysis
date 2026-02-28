"""
Solver comparison framework.

Run multiple solvers on the same case and collect results for comparison.
Designed for comparing tangential velocity (Vt), pressure coefficient (Cp),
and other surface quantities across different solver formulations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from core.io.case import Case
from postprocessing.surface import SurfaceData, SurfaceDataExtractor
from solvers.base import Solver


# ── Solver specification ────────────────────────────────────────────────────

# Mapping from short names to legacy solver type strings recognised by SolverConfig
SOLVER_ALIASES: Dict[str, str] = {
    "constant": "constant_source",
    "constant_source": "constant_source",
    "linear": "linear_source",
    "linear_source": "linear_source",
}


def _resolve_solver_type(name: str) -> str:
    """Resolve a human-friendly name to a SolverConfig-compatible legacy type."""
    key = name.strip().lower()
    if key in SOLVER_ALIASES:
        return SOLVER_ALIASES[key]
    # Pass through for future solver types registered directly
    return name


# ── Result containers ───────────────────────────────────────────────────────

@dataclass
class SolverResult:
    """Result from a single solver run.

    Attributes:
        solver_type: Legacy solver type string (e.g. "constant_source").
        label: Human-readable label for plots.
        solver: The solved Solver instance.
        surface: Extracted SurfaceData with Vt, Cp, coordinates.
        solve_time_s: Wall-clock solve time in seconds.
    """
    solver_type: str
    label: str
    solver: Solver
    surface: SurfaceData
    solve_time_s: float = 0.0


@dataclass
class ComparisonResult:
    """Aggregated comparison across multiple solvers.

    Attributes:
        case: The originating Case object.
        results: List of individual SolverResult entries.
        metrics: Per-pair error metrics (populated by compute_metrics).
    """
    case: Case
    results: List[SolverResult] = field(default_factory=list)
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # ── convenience accessors ───────────────────────────────────────────

    @property
    def labels(self) -> List[str]:
        return [r.label for r in self.results]

    @property
    def surface_list(self) -> List[SurfaceData]:
        return [r.surface for r in self.results]

    @property
    def x_list(self) -> List[NDArray]:
        return [r.surface.x for r in self.results]

    @property
    def y_list(self) -> List[NDArray]:
        return [r.surface.y for r in self.results]

    @property
    def vt_list(self) -> List[NDArray]:
        return [r.surface.Vt for r in self.results]

    @property
    def cp_list(self) -> List[NDArray]:
        return [r.surface.Cp for r in self.results]

    # ── metrics ─────────────────────────────────────────────────────────

    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute pairwise error metrics between all solver results.

        Metrics computed (for Vt):
            L_inf: max absolute difference
            L2:    RMS of absolute difference
            MAE:   mean absolute error
            rel_L2: RMS / mean(|reference|) (relative RMS %)

        The first result is treated as the reference. Metrics are computed
        for every other result vs. the first.

        Returns:
            Dict mapping ``"<label_a> vs <label_b>"`` to metric dict.
        """
        if len(self.results) < 2:
            return {}

        ref = self.results[0]
        metrics: Dict[str, Dict[str, float]] = {}

        for other in self.results[1:]:
            key = f"{other.label} vs {ref.label}"
            diff_vt = other.surface.Vt - ref.surface.Vt
            diff_cp = other.surface.Cp - ref.surface.Cp

            ref_vt_mean = np.mean(np.abs(ref.surface.Vt))
            ref_cp_mean = np.mean(np.abs(ref.surface.Cp))

            metrics[key] = {
                "Vt_Linf": float(np.max(np.abs(diff_vt))),
                "Vt_L2": float(np.sqrt(np.mean(diff_vt ** 2))),
                "Vt_MAE": float(np.mean(np.abs(diff_vt))),
                "Vt_rel_L2_pct": float(
                    np.sqrt(np.mean(diff_vt ** 2)) / ref_vt_mean * 100
                ) if ref_vt_mean > 1e-12 else float("inf"),
                "Cp_Linf": float(np.max(np.abs(diff_cp))),
                "Cp_L2": float(np.sqrt(np.mean(diff_cp ** 2))),
                "Cp_MAE": float(np.mean(np.abs(diff_cp))),
                "Cp_rel_L2_pct": float(
                    np.sqrt(np.mean(diff_cp ** 2)) / ref_cp_mean * 100
                ) if ref_cp_mean > 1e-12 else float("inf"),
            }

        self.metrics = metrics
        return metrics


# ── Runner ──────────────────────────────────────────────────────────────────

class SolverComparisonRunner:
    """Run multiple solvers on the same case and collect results.

    Usage::

        runner = SolverComparisonRunner(case)
        result = runner.run(["constant_source", "linear_source"])

        # Or with explicit labels
        result = runner.run(
            solver_types=["constant_source", "linear_source"],
            labels=["Constant Source", "Linear Source"],
        )

        # Access results
        for r in result.results:
            print(f"{r.label}: Vt range [{r.surface.Vt.min():.4f}, {r.surface.Vt.max():.4f}]")
    """

    def __init__(self, case: Case) -> None:
        self.case = case

    def run(
        self,
        solver_types: Sequence[str],
        labels: Optional[Sequence[str]] = None,
        *,
        verbose: bool = True,
    ) -> ComparisonResult:
        """Run all specified solvers and collect results.

        Args:
            solver_types: Solver type names (e.g. ["constant_source", "linear_source"]).
                Accepts short aliases like "constant" / "linear".
            labels: Optional human-readable labels for each solver.
                If None, labels are derived from solver_types.
            verbose: Print progress information.

        Returns:
            ComparisonResult with all solver results and computed metrics.
        """
        import time

        resolved = [_resolve_solver_type(s) for s in solver_types]
        if labels is None:
            labels = [self._auto_label(s) for s in resolved]

        if len(resolved) != len(labels):
            raise ValueError(
                f"Number of solver_types ({len(resolved)}) must match "
                f"number of labels ({len(labels)})"
            )

        comparison = ComparisonResult(case=self.case)

        for solver_type, label in zip(resolved, labels):
            if verbose:
                print(f"\n{'='*60}")
                print(f"  Solver: {label} ({solver_type})")
                print(f"  Mesh:   {self.case.num_panels} panels")
                print(f"{'='*60}")

            # Create and solve
            t0 = time.perf_counter()
            solver = self.case.create_solver(solver_type=solver_type)
            solver.solve()
            elapsed = time.perf_counter() - t0

            if verbose:
                print(f"  Solved in {elapsed:.3f}s")
                print(f"  Vt range: [{solver.Vt.min():.4f}, {solver.Vt.max():.4f}]")
                print(f"  Cp range: [{solver.Cp.min():.4f}, {solver.Cp.max():.4f}]")

            # Extract surface data
            extractor = SurfaceDataExtractor(self.case.mesh, solver)
            surface = extractor.extract(arc_length=True)

            comparison.results.append(
                SolverResult(
                    solver_type=solver_type,
                    label=label,
                    solver=solver,
                    surface=surface,
                    solve_time_s=elapsed,
                )
            )

        # Compute comparison metrics
        if len(comparison.results) >= 2:
            comparison.compute_metrics()
            if verbose:
                self._print_metrics(comparison)

        return comparison

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _auto_label(solver_type: str) -> str:
        """Generate a readable label from a solver type string."""
        return solver_type.replace("_", " ").title()

    @staticmethod
    def _print_metrics(comparison: ComparisonResult) -> None:
        """Pretty-print comparison metrics."""
        print(f"\n{'='*60}")
        print("  Comparison Metrics")
        print(f"{'='*60}")
        for pair, m in comparison.metrics.items():
            print(f"\n  {pair}:")
            print(f"    Vt  L_inf: {m['Vt_Linf']:.6f}  "
                  f"RMS: {m['Vt_L2']:.6f}  "
                  f"MAE: {m['Vt_MAE']:.6f}  "
                  f"rel RMS: {m['Vt_rel_L2_pct']:.2f}%")
            print(f"    Cp  L_inf: {m['Cp_Linf']:.6f}  "
                  f"RMS: {m['Cp_L2']:.6f}  "
                  f"MAE: {m['Cp_MAE']:.6f}  "
                  f"rel RMS: {m['Cp_rel_L2_pct']:.2f}%")
