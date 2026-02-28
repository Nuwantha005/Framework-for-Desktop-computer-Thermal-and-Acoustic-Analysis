#!/usr/bin/env python3
"""
Boundary Layer Analysis Demo
=============================

Run boundary-layer analysis on a solved panel-method case.

Identifies stagnation points via ``n · V∞``, splits the body into upper and
lower surface streamlines, and computes BL quantities (cf, δ*, θ, H) along
each path using one or more velocity profiles.

Usage::

    python demos/demo_boundary_layer.py cases/rounded_square
    python demos/demo_boundary_layer.py cases/rounded_square --solver-type linear_source
    python demos/demo_boundary_layer.py cases/rounded_square --profiles blasius thwaites
    python demos/demo_boundary_layer.py cases/rounded_square --show-plots
"""

import sys
import argparse
from pathlib import Path

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.io.case_loader import CaseLoader
from solvers.boundary_layer.runner import BoundaryLayerRunner
from visualization.bl_plots import (
    plot_bl_lines_multi,
    plot_bl_two_sides,
    plot_bl_envelope,
    plot_bl_envelope_comparison,
    plot_bl_comparison,
)


def _print_summary(bl) -> None:
    """Print per-side, per-profile BL summary."""
    for side_name, path in bl.sides.items():
        n_panels = len(path.panel_indices)
        print(f"\n  === {side_name.upper()} SIDE ({n_panels} panels, "
              f"s_max = {path.s[-1]:.4f} m) ===")

        for name, res in path.results.items():
            valid = ~np.isnan(res.theta)
            n_valid = int(valid.sum())
            n_total = len(res.theta)
            print(f"    [{name}]  {n_valid}/{n_total} stations valid, "
                  f"converged={res.converged}")
            if n_valid > 0:
                print(f"      θ:  [{np.nanmin(res.theta):.3e}, "
                      f"{np.nanmax(res.theta):.3e}]")
                print(f"      cf: [{np.nanmin(res.cf):.3e}, "
                      f"{np.nanmax(res.cf):.3e}]")
                print(f"      δ*: [{np.nanmin(res.delta_star):.3e}, "
                      f"{np.nanmax(res.delta_star):.3e}]")
                print(f"      H:  [{np.nanmin(res.H):.2f}, "
                      f"{np.nanmax(res.H):.2f}]")

            tr = path.transitions.get(name)
            if tr is not None:
                if tr.transition_s is not None:
                    print(f"      Transition: s = {tr.transition_s:.4f} "
                          f"({tr.criterion_name})")
                else:
                    print(f"      No transition ({tr.criterion_name})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run boundary layer analysis on a panel-method case",
    )
    parser.add_argument("case_dir", type=Path,
                        help="Path to case directory (e.g. cases/rounded_square)")
    parser.add_argument("--solver-type", type=str, default=None,
                        help="Panel solver type (default: from case config)")
    parser.add_argument("--mesh-level", type=int, default=-1,
                        help="Mesh refinement level index (default: finest)")
    parser.add_argument("--profiles", nargs="+", default=None,
                        help="BL profiles (e.g. blasius thwaites)")
    parser.add_argument("--transition", type=str, default=None,
                        choices=["michel", "en"],
                        help="Transition prediction model")
    parser.add_argument("--nu", type=float, default=None,
                        help="Kinematic viscosity override [m²/s]")
    parser.add_argument("--envelope-scale", type=float, default=None,
                        help="Envelope displacement scale factor")
    parser.add_argument("--show-plots", action="store_true",
                        help="Display plots interactively")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: <case>/out/boundary_layer)")
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    if not case_dir.exists():
        print(f"Error: Case directory not found: {case_dir}")
        return 1

    # ------------------------------------------------------------------
    # 1. Load case and solve panel method
    # ------------------------------------------------------------------
    print(f"Loading case: {case_dir}")
    case = CaseLoader.load_case(case_dir, mesh_level_index=args.mesh_level)
    print(f"  Case: {case.name}, {case.mesh.num_panels} panels")

    solver = case.create_solver(solver_type=args.solver_type)
    print(f"  Solver: {solver.__class__.__name__}")
    print("  Solving panel method ...")
    solver.solve()
    print("  Done.\n")

    # ------------------------------------------------------------------
    # 2. Run boundary layer analysis
    # ------------------------------------------------------------------
    runner = BoundaryLayerRunner(case, solver)
    bl_cfg = case.config.boundary_layer
    profiles = args.profiles or list(bl_cfg.profiles)
    transition = args.transition or bl_cfg.transition_model
    envelope_scale = args.envelope_scale or bl_cfg.envelope_scale

    print(f"  BL profiles: {profiles}")
    if transition:
        print(f"  Transition model: {transition}")

    bl = runner.run(
        profiles=profiles,
        nu=args.nu,
        transition_model=transition,
    )

    print(f"\n  ν = {bl.nu:.3e} m²/s")
    _print_summary(bl)

    # ------------------------------------------------------------------
    # 3. Generate plots
    # ------------------------------------------------------------------
    import matplotlib
    if not args.show_plots:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = args.output_dir or (case.output_dir / "boundary_layer")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving plots to: {output_dir}")

    # Two-sided line plots (upper | lower)
    plot_bl_two_sides(
        bl,
        title=f"Boundary Layer — {case.name}",
        output_path=output_dir / "bl_lines.png",
    )
    plt.close("all")
    print("  ✓ bl_lines.png")

    # Per-side multi-panel plots
    for side_name, path in bl.sides.items():
        plot_bl_lines_multi(
            path,
            title=f"{side_name.capitalize()} side — {case.name}",
            output_path=output_dir / f"bl_lines_{side_name}.png",
        )
        plt.close("all")
    print("  ✓ bl_lines_upper.png, bl_lines_lower.png")

    # Envelope comparison (cf and δ*)
    for qty, label in [("cf", "cf"), ("delta_star", "δ*")]:
        plot_bl_envelope_comparison(
            bl,
            quantity=qty,
            scale=envelope_scale,
            title=f"{label} envelope — {case.name}",
            output_path=output_dir / f"bl_{qty}_envelope.png",
        )
        plt.close("all")
    print("  ✓ bl_cf_envelope.png, bl_delta_star_envelope.png")

    # Full comparison figure
    plot_bl_comparison(
        bl,
        envelope_scale=envelope_scale,
        title=f"Boundary Layer Comparison — {case.name}",
        output_path=output_dir / "bl_comparison.png",
    )
    plt.close("all")
    print("  ✓ bl_comparison.png")

    # Individual profile envelope plots
    for name in bl.profile_names:
        safe = name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        plot_bl_envelope(
            bl,
            quantity="cf",
            profile_name=name,
            scale=envelope_scale,
            title=f"cf envelope — {name}",
            output_path=output_dir / f"bl_cf_envelope_{safe}.png",
        )
        plt.close("all")
    print("  ✓ Individual profile envelopes")

    if args.show_plots:
        plt.show()

    print("\nBoundary layer analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
