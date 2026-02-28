"""
Solver comparison framework.

Run multiple panel method solvers on the same case, optionally include
OpenFOAM CFD reference data, compute pairwise error metrics, and rank
solvers by accuracy against the reference.

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
    "vortex": "linear_vortex",
    "linear_vortex": "linear_vortex",
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
    """Result from a single solver run or external reference dataset.

    Attributes:
        solver_type: Legacy solver type string (e.g. "constant_source") or "openfoam".
        label: Human-readable label for plots.
        solver: The solved Solver instance (None for external references).
        surface: Extracted SurfaceData with Vt, Cp, coordinates.
        solve_time_s: Wall-clock solve time in seconds.
        is_reference: True if this is a CFD reference (e.g. OpenFOAM).
    """
    solver_type: str
    label: str
    solver: Optional[Solver]
    surface: SurfaceData
    solve_time_s: float = 0.0
    is_reference: bool = False


@dataclass
class ComparisonResult:
    """Aggregated comparison across multiple solvers and optional reference.

    When a reference result is present (``is_reference=True``), metrics are
    computed for every panel solver against the reference instead of the
    first solver. A ``ranking`` dict orders solvers by Vt relative-L2.

    Attributes:
        case: The originating Case object.
        results: List of individual SolverResult entries.
        metrics: Per-pair error metrics (populated by compute_metrics).
        ranking: Solver labels sorted by Vt_rel_L2_pct ascending (best first).
    """
    case: Case
    results: List[SolverResult] = field(default_factory=list)
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    ranking: List[Tuple[str, float]] = field(default_factory=list)

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

    @property
    def reference(self) -> Optional[SolverResult]:
        """Return the reference result (OpenFOAM) if present."""
        for r in self.results:
            if r.is_reference:
                return r
        return None

    @property
    def solver_results(self) -> List[SolverResult]:
        """Return only panel-method solver results (non-reference)."""
        return [r for r in self.results if not r.is_reference]

    # ── metrics ─────────────────────────────────────────────────────────

    @staticmethod
    def _compute_pair_metrics(
        ref_vt: NDArray, ref_cp: NDArray,
        other_vt: NDArray, other_cp: NDArray,
    ) -> Dict[str, float]:
        """Compute error metrics for a single pair of Vt/Cp arrays."""
        diff_vt = other_vt - ref_vt
        diff_cp = other_cp - ref_cp
        ref_vt_mean = np.mean(np.abs(ref_vt))
        ref_cp_mean = np.mean(np.abs(ref_cp))

        return {
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

    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute pairwise error metrics.

        When an OpenFOAM reference is present, every panel-method solver is
        compared against the reference (with interpolation to the solver's
        arc-length grid). Otherwise the first result is the reference and
        remaining results are compared pairwise against it.

        Populates ``self.ranking`` with (label, Vt_rel_L2_pct) tuples
        sorted best-first when a reference is present.

        Returns:
            Dict mapping ``"<label_a> vs <label_b>"`` to metric dict.
        """
        if len(self.results) < 2:
            return {}

        ref_result = self.reference
        metrics: Dict[str, Dict[str, float]] = {}

        if ref_result is not None:
            # Compare each panel solver against the OpenFOAM reference
            for sr in self.solver_results:
                key = f"{sr.label} vs {ref_result.label}"
                # Interpolate reference to solver's arc-length grid
                ref_vt_interp, ref_cp_interp = self._interpolate_reference(
                    ref_result.surface, sr.surface
                )
                metrics[key] = self._compute_pair_metrics(
                    ref_vt_interp, ref_cp_interp,
                    sr.surface.Vt, sr.surface.Cp,
                )
        else:
            # No reference — use first result as baseline
            ref = self.results[0]
            for other in self.results[1:]:
                key = f"{other.label} vs {ref.label}"
                metrics[key] = self._compute_pair_metrics(
                    ref.surface.Vt, ref.surface.Cp,
                    other.surface.Vt, other.surface.Cp,
                )

        self.metrics = metrics

        # Build ranking (best Vt relative-L2 first)
        self.ranking = sorted(
            [(pair.split(" vs ")[0], m["Vt_rel_L2_pct"]) for pair, m in metrics.items()],
            key=lambda t: t[1],
        )

        return metrics

    @staticmethod
    def _interpolate_reference(
        ref_surface: SurfaceData, target_surface: SurfaceData
    ) -> Tuple[NDArray, NDArray]:
        """Interpolate reference Vt/Cp onto the target's arc-length grid.

        Normalises both arc-length arrays to [0, 1] so that different point
        counts and perimeter lengths are handled correctly. Duplicate
        arc-length values in the reference are removed before interpolation.
        """
        from scipy.interpolate import interp1d

        s_ref = ref_surface.s
        s_target = target_surface.s

        # Normalise to [0, 1]
        s_ref_norm = (s_ref - s_ref.min()) / (s_ref.max() - s_ref.min() + 1e-30)
        s_target_norm = (s_target - s_target.min()) / (s_target.max() - s_target.min() + 1e-30)

        # Remove duplicate arc-length values (e.g. wrap-around start/end)
        _, unique_idx = np.unique(s_ref_norm, return_index=True)
        unique_idx = np.sort(unique_idx)  # Preserve original order
        s_ref_unique = s_ref_norm[unique_idx]
        vt_ref_unique = ref_surface.Vt[unique_idx]
        cp_ref_unique = ref_surface.Cp[unique_idx]

        vt_interp = interp1d(
            s_ref_unique, vt_ref_unique, kind="linear",
            bounds_error=False, fill_value="extrapolate",
        )(s_target_norm)

        cp_interp = interp1d(
            s_ref_unique, cp_ref_unique, kind="linear",
            bounds_error=False, fill_value="extrapolate",
        )(s_target_norm)

        return np.asarray(vt_interp), np.asarray(cp_interp)


# ── OpenFOAM reference extraction ──────────────────────────────────────────

def extract_openfoam_reference(
    case: Case,
    of_case_dir: Path,
    *,
    verbose: bool = True,
) -> SolverResult:
    """Extract OpenFOAM surface data and wrap as a SolverResult.

    Uses the same extraction pipeline as ``compare_surface.py``:
    ``OpenFOAMRunner.run_post_process()`` followed by
    ``OpenFOAMSurfaceExtractor.extract()`` for each wall patch.

    Args:
        case: Panel method Case (needed for v_inf, density, component names).
        of_case_dir: Absolute path to the OpenFOAM case directory.
        verbose: Print progress.

    Returns:
        SolverResult with ``is_reference=True`` and source ``"openfoam"``.
    """
    from validation.adapters.openfoam import OpenFOAMSurfaceExtractor, OpenFOAMRunner
    from validation.adapters.openfoam.foamlib_generator import sanitize_name

    v_inf = case.v_inf
    density = case.config.fluid.density

    # Run postProcess to make sure VTP files exist
    if verbose:
        print(f"\n{'='*60}")
        print(f"  OpenFOAM Reference: {of_case_dir.name}")
        print(f"{'='*60}")
        print("  Running postProcess...")

    runner = OpenFOAMRunner(of_case_dir, verbose=verbose)
    result = runner.run_post_process(fields=["U", "p"])
    if not result.success:
        if verbose:
            print(f"  WARNING: postProcess failed: {result.stderr}")
            print("  Attempting to continue with existing data...")
    elif verbose:
        print("  postProcess complete")

    # Determine patch names
    if case.num_components == 1:
        patch_names = [sanitize_name(case.name)]
    else:
        patch_names = [sanitize_name(comp.name) for comp in case.scene.components]

    if verbose:
        print(f"  Patch names: {patch_names}")

    # Create extractor
    try:
        extractor = OpenFOAMSurfaceExtractor(of_case_dir, time_idx=-1)
    except FileNotFoundError as e:
        processor_dirs = list(of_case_dir.glob("processor*"))
        if processor_dirs:
            raise RuntimeError(
                f"OpenFOAM case appears to be parallel ({len(processor_dirs)} processor dirs). "
                f"Try: cd {of_case_dir} && reconstructPar"
            )
        raise RuntimeError(f"Failed to create OpenFOAM extractor: {e}")

    # Extract each patch
    surface_parts: List[SurfaceData] = []
    for comp_id, patch_name in enumerate(patch_names):
        if verbose:
            print(f"  Extracting patch: {patch_name}...")
        try:
            data = extractor.extract(
                patch_name=patch_name,
                reference_pressure=0.0,
                density=density,
                v_inf=v_inf,
            )
            data.component_id = np.full(len(data.x), comp_id, dtype=np.int32)
            surface_parts.append(data)
            if verbose:
                print(f"    Points: {len(data.x)}, "
                      f"Vt: [{data.Vt.min():.4f}, {data.Vt.max():.4f}], "
                      f"Cp: [{data.Cp.min():.4f}, {data.Cp.max():.4f}]")
        except Exception as e:
            if verbose:
                print(f"    WARNING: Could not extract patch {patch_name}: {e}")
            continue

    if not surface_parts:
        raise RuntimeError("No surface data extracted from OpenFOAM")

    # Concatenate patches
    if len(surface_parts) == 1:
        of_surface = surface_parts[0]
    else:
        of_surface = SurfaceData(
            x=np.concatenate([d.x for d in surface_parts]),
            y=np.concatenate([d.y for d in surface_parts]),
            s=np.concatenate([d.s for d in surface_parts]),
            Vt=np.concatenate([d.Vt for d in surface_parts]),
            Vn=np.concatenate([d.Vn for d in surface_parts])
            if surface_parts[0].Vn is not None else None,
            Cp=np.concatenate([d.Cp for d in surface_parts]),
            component_id=np.concatenate([d.component_id for d in surface_parts]),
            source="openfoam",
        )

    return SolverResult(
        solver_type="openfoam",
        label="OpenFOAM",
        solver=None,
        surface=of_surface,
        solve_time_s=0.0,
        is_reference=True,
    )


# ── Runner ──────────────────────────────────────────────────────────────────

class SolverComparisonRunner:
    """Run multiple solvers on the same case and collect results.

    Usage::

        runner = SolverComparisonRunner(case)
        result = runner.run(["constant_source", "linear_source"])

        # With OpenFOAM reference
        result = runner.run(
            ["constant", "linear"],
            of_case_dir=Path("cases/rounded_square/of_case/cases/level_4"),
        )

        # Access results and ranking
        for label, rel_pct in result.ranking:
            print(f"  {label}: {rel_pct:.2f}% relative Vt error")
    """

    def __init__(self, case: Case) -> None:
        self.case = case

    def run(
        self,
        solver_types: Sequence[str],
        labels: Optional[Sequence[str]] = None,
        *,
        of_case_dir: Optional[Path] = None,
        of_label: str = "OpenFOAM",
        verbose: bool = True,
    ) -> ComparisonResult:
        """Run all specified solvers and collect results.

        Args:
            solver_types: Solver type names (e.g. ["constant_source", "linear_source"]).
                Accepts short aliases like "constant" / "linear".
            labels: Optional human-readable labels for each solver.
                If None, labels are derived from solver_types.
            of_case_dir: Path to an OpenFOAM case directory. If provided,
                OpenFOAM surface data is extracted and used as the reference
                for metric computation and ranking.
            of_label: Label for the OpenFOAM reference in plots.
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

        # ── Extract OpenFOAM reference (if requested) ───────────────────
        # When an OF case is provided we also discover the STL reference
        # geometry so that panel-method surface extraction uses the same
        # arc-length origin (landmark="min_x") as the OF extractor.
        reference_stl: Optional[str] = None
        if of_case_dir is not None:
            resolved_of = Path(of_case_dir).resolve()
            of_result = extract_openfoam_reference(
                self.case, resolved_of, verbose=verbose,
            )
            of_result.label = of_label
            comparison.results.append(of_result)

            # Locate STL used by the OF case for geometry-projected arc length
            stl_dir = resolved_of / "constant" / "triSurface"
            stl_files = list(stl_dir.glob("*.stl")) if stl_dir.exists() else []
            if stl_files:
                reference_stl = str(stl_files[0])
                if verbose:
                    print(f"  Reference STL for arc length: {Path(reference_stl).name}")

        # ── Run panel-method solvers ────────────────────────────────────
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

            # Extract surface data — use STL reference geometry when available
            # so that the arc-length origin matches the OF extractor's
            # normalize_arc_length(landmark="min_x") convention.
            extractor = SurfaceDataExtractor(self.case.mesh, solver)
            surface = extractor.extract(
                arc_length=True,
                reference_geometry=reference_stl,
            )

            # After normalize_arc_length wraps s via modulo, data is no
            # longer monotonic in s.  Sort so line plots are continuous.
            if surface.s is not None:
                sort_idx = np.argsort(surface.s)
                surface = SurfaceData(
                    x=surface.x[sort_idx],
                    y=surface.y[sort_idx],
                    s=surface.s[sort_idx],
                    Vt=surface.Vt[sort_idx],
                    Vn=surface.Vn[sort_idx] if surface.Vn is not None else None,
                    Cp=surface.Cp[sort_idx],
                    component_id=surface.component_id[sort_idx] if surface.component_id is not None else None,
                    source=surface.source,
                )

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
        """Pretty-print comparison metrics and ranking."""
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

        if comparison.ranking:
            print(f"\n{'='*60}")
            print("  Solver Ranking (by Vt relative RMS vs reference)")
            print(f"{'='*60}")
            for rank, (label, rel_pct) in enumerate(comparison.ranking, 1):
                marker = " <-- best" if rank == 1 else ""
                print(f"    #{rank}  {label:25s}  {rel_pct:8.3f}%{marker}")
