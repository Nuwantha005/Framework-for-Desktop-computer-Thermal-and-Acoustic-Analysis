#!/usr/bin/env python3
"""
Boundary Layer Analysis Demo
=============================

Run boundary-layer analysis on a solved panel-method case.

Identifies stagnation points via ``n · V∞``, splits the body into upper and
lower surface streamlines, and computes BL quantities (cf, δ*, θ, H) along
each path using one or more velocity profiles.

When ``--reconstruct`` is given, also generates velocity-field visualizations:
s-y contour plots, normalised y/δ contours, and wrapped velocity envelopes.

When ``--compare-fluent`` is given, loads Fluent CFD data from the case's
fluent_case/export/viscous_bl/ directory and generates comparison plots.

Usage::

    python demos/demo_boundary_layer.py cases/rounded_square
    python demos/demo_boundary_layer.py cases/rounded_square --solver-type linear_source
    python demos/demo_boundary_layer.py cases/rounded_square --profiles blasius thwaites
    python demos/demo_boundary_layer.py cases/rounded_square --reconstruct
    python demos/demo_boundary_layer.py cases/rounded_square --compare-fluent
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
    plot_bl_velocity_contour_two_sides,
    plot_bl_velocity_contour_normalized_two_sides,
    plot_bl_velocity_envelope_two_sides,
    plot_bl_fluent_comparison,
    plot_bl_fluent_comparison_two_sides,
    plot_bl_wall_comparison,
    plot_bl_velocity_envelope_comparison,
    plot_bl_velocity_contour_normalized_comparison,
    plot_bl_comparison_report,
    plot_bl_fluent_envelope_side_by_side,
    plot_bl_fluent_contour_side_by_side,
    plot_bl_fluent_contour_normalized_side_by_side,
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

            fld = path.fields.get(name)
            if fld is not None:
                print(f"      Reconstructed field: "
                      f"{fld.u.shape[0]} stations × {fld.u.shape[1]} y-points, "
                      f"δ_max = {np.nanmax(fld.delta):.3e} m")


def _print_comparison_metrics(result) -> None:
    """Print comparison error metrics."""
    print("\n  === COMPARISON METRICS ===")

    if result.wall_metrics:
        for side, metrics in result.wall_metrics.items():
            print(f"\n    {side.upper()} SIDE (wall quantities):")
            for qty, m in metrics.items():
                print(f"      {qty}: RMS={m.RMS:.4e}, MAE={m.MAE:.4e}, "
                      f"rel_L2={m.relative_L2:.2%} ({m.n_points} pts)")

    if result.velocity_metrics:
        print("\n    VELOCITY FIELD:")
        for side, metrics in result.velocity_metrics.items():
            print(f"      {side}: ", end="")
            for qty, m in metrics.items():
                print(f"{qty} RMS={m.RMS:.4e} ", end="")
            print()


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
    parser.add_argument("--reconstruct", action="store_true",
                        help="Run velocity-field reconstruction and generate "
                             "contour/envelope plots")
    parser.add_argument("--compare-fluent", action="store_true",
                        help="Compare with Fluent CFD data (requires "
                             "fluent_case/export/viscous_bl/ data)")
    parser.add_argument("--compare-profile", type=str, default=None,
                        help="BL profile to use for Fluent comparison. "
                             "If not specified, compares all profiles with reconstruction.")
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

    # Force reconstruction if comparing with Fluent
    reconstruct = args.reconstruct or args.compare_fluent

    print(f"  BL profiles: {profiles}")
    if transition:
        print(f"  Transition model: {transition}")
    if reconstruct:
        print("  Velocity-field reconstruction: enabled")

    bl = runner.run(
        profiles=profiles,
        nu=args.nu,
        transition_model=transition,
        reconstruct=reconstruct,
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
    print("  + bl_lines.png")

    # Per-side multi-panel plots
    for side_name, path in bl.sides.items():
        plot_bl_lines_multi(
            path,
            title=f"{side_name.capitalize()} side — {case.name}",
            output_path=output_dir / f"bl_lines_{side_name}.png",
        )
        plt.close("all")
    print("  + bl_lines_upper.png, bl_lines_lower.png")

    # Envelope comparison (cf and δ*)
    for qty, label in [("cf", "cf"), ("delta_star", "delta_star")]:
        plot_bl_envelope_comparison(
            bl,
            quantity=qty,
            scale=envelope_scale,
            title=f"{label} envelope — {case.name}",
            output_path=output_dir / f"bl_{qty}_envelope.png",
        )
        plt.close("all")
    print("  + bl_cf_envelope.png, bl_delta_star_envelope.png")

    # Full comparison figure
    plot_bl_comparison(
        bl,
        envelope_scale=envelope_scale,
        title=f"Boundary Layer Comparison — {case.name}",
        output_path=output_dir / "bl_comparison.png",
    )
    plt.close("all")
    print("  + bl_comparison.png")

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
    print("  + Individual profile envelopes")

    # ------------------------------------------------------------------
    # 4. Velocity-field plots (only when --reconstruct is given)
    # ------------------------------------------------------------------
    if reconstruct:
        for name in bl.profile_names:
            safe = name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
            field_u = bl.upper.fields.get(name)
            field_l = bl.lower.fields.get(name)
            if field_u is None or field_l is None:
                print(f"  (skipping velocity plots for {name} — "
                      f"reconstruction unavailable)")
                continue

            # s-y velocity contour (both sides)
            plot_bl_velocity_contour_two_sides(
                field_u, field_l, cmap="viridis",
                title=f"BL velocity contour — {name} — {case.name}",
                output_path=output_dir / f"bl_vel_contour_{safe}.png",
            )
            plt.close("all")
            print(f"  + bl_vel_contour_{safe}.png")

            # Normalised s-(y/δ) contour
            plot_bl_velocity_contour_normalized_two_sides(
                field_u, field_l, cmap="viridis",
                title=f"Normalised BL velocity — {name} — {case.name}",
                output_path=output_dir / f"bl_vel_normalised_{safe}.png",
            )
            plt.close("all")
            print(f"  + bl_vel_normalised_{safe}.png")

            # Wrapped velocity envelope
            plot_bl_velocity_envelope_two_sides(
                field_u, field_l, bl,
                scale=envelope_scale,
                cmap="viridis",
                title=f"Velocity envelope — {name} — {case.name}",
                output_path=output_dir / f"bl_vel_envelope_{safe}.png",
            )
            plt.close("all")
            print(f"  + bl_vel_envelope_{safe}.png")

    # ------------------------------------------------------------------
    # 5. Fluent comparison (when --compare-fluent is given)
    # ------------------------------------------------------------------
    if args.compare_fluent:
        print("\n  --- Fluent Comparison ---")

        from validation.adapters.fluent import BLComparisonRunner

        # Create fluent comparison output directory
        fluent_dir = output_dir / "fluent_comparison"
        fluent_dir.mkdir(parents=True, exist_ok=True)

        # Determine which profiles to compare
        if args.compare_profile:
            # Single profile specified
            compare_profiles = [args.compare_profile]
        else:
            # All available profiles with reconstruction
            compare_profiles = list(bl.upper.fields.keys())

        for profile_name in compare_profiles:
            comp_runner = BLComparisonRunner(case, bl)
            comp_result = comp_runner.run(profile_name=profile_name)

            if not comp_result.has_fluent_data:
                print("  Warning: Fluent data not available")
                print("  Expected: fluent_case/export/viscous_bl/{filed_data,wall_data}")
                break

            # Use normalized profile name for file naming
            pname = comp_result.profile_name
            safe_pname = pname.lower().replace(" ", "_").replace("-", "_")

            if profile_name == compare_profiles[0]:
                print("  Fluent data loaded successfully")

            print(f"\n  Profile: {pname}")
            _print_comparison_metrics(comp_result)

            # Velocity field comparison (two sides)
            plot_bl_fluent_comparison_two_sides(
                bl, comp_result,
                profile_name=pname,
                title=f"Velocity difference — {pname} — {case.name}",
                output_path=fluent_dir / f"bl_fluent_velocity_diff_{safe_pname}.png",
            )
            plt.close("all")
            print(f"  + fluent_comparison/bl_fluent_velocity_diff_{safe_pname}.png")

            # Wall quantities comparison
            plot_bl_wall_comparison(
                bl, comp_result.fluent_result,
                quantities=["Ue", "Cf", "delta"],
                profile_name=pname,
                title=f"Wall Quantities — {pname} — {case.name}",
                output_path=fluent_dir / f"bl_fluent_wall_{safe_pname}.png",
            )
            plt.close("all")
            print(f"  + fluent_comparison/bl_fluent_wall_{safe_pname}.png")

            # Combined Envelope comparison plot
            plot_bl_velocity_envelope_comparison(
                bl, comp_result,
                profile_name=pname,
                title=f"Velocity difference envelope — {pname} vs Fluent — {case.name}",
                output_path=fluent_dir / f"bl_fluent_envelope_{safe_pname}.png",
            )
            plt.close("all")
            print(f"  + fluent_comparison/bl_fluent_envelope_{safe_pname}.png")

            # Individual side comparisons
            for side in ["upper", "lower"]:
                field = bl.sides[side].fields.get(pname)
                fluent_field = comp_result.sides[side]
                if field is not None and fluent_field is not None:
                    plot_bl_fluent_comparison(
                        field, fluent_field,
                        title=f"{side.capitalize()} — {pname} vs Fluent — {case.name}",
                        output_path=fluent_dir / f"bl_fluent_{side}_{safe_pname}.png",
                    )
                    plt.close("all")
                    print(f"  + fluent_comparison/bl_fluent_{side}_{safe_pname}.png")

                    # Normalized comparison plot
                    plot_bl_velocity_contour_normalized_comparison(
                        field, fluent_field,
                        title=f"{side.capitalize()} normalized — {pname} vs Fluent — {case.name}",
                        output_path=fluent_dir / f"bl_fluent_normalized_{side}_{safe_pname}.png",
                    )
                    plt.close("all")
                    print(f"  + fluent_comparison/bl_fluent_normalized_{side}_{safe_pname}.png")

            # Comprehensive metrics report
            plot_bl_comparison_report(
                comp_result,
                title=f"Comparison Report — {pname} — {case.name}",
                output_path=fluent_dir / f"bl_fluent_report_{safe_pname}.png",
            )
            plt.close("all")
            print(f"  + fluent_comparison/bl_fluent_report_{safe_pname}.png")

            # --- Side-by-side Absolute Velocity Comparisons ---
            
            # Side-by-side Envelope
            plot_bl_fluent_envelope_side_by_side(
                bl, comp_result,
                profile_name=pname,
                title=f"Absolute Velocity Envelope — {pname} vs Fluent — {case.name}",
                output_path=fluent_dir / f"bl_fluent_envelope_abs_{safe_pname}.png",
            )
            plt.close("all")
            print(f"  + fluent_comparison/bl_fluent_envelope_abs_{safe_pname}.png")

            for side in ["upper", "lower"]:
                field = bl.sides[side].fields.get(pname)
                fluent_field = comp_result.sides[side]
                if field is not None and fluent_field is not None:
                    # Side-by-side Contour
                    plot_bl_fluent_contour_side_by_side(
                        field, fluent_field,
                        title=f"{side.capitalize()} Absolute Velocity — {pname} vs Fluent — {case.name}",
                        output_path=fluent_dir / f"bl_fluent_contour_abs_{side}_{safe_pname}.png",
                    )
                    plt.close("all")
                    print(f"  + fluent_comparison/bl_fluent_contour_abs_{side}_{safe_pname}.png")

                    # Side-by-side Normalized Contour
                    plot_bl_fluent_contour_normalized_side_by_side(
                        field, fluent_field,
                        title=f"{side.capitalize()} Normalized Velocity — {pname} vs Fluent — {case.name}",
                        output_path=fluent_dir / f"bl_fluent_normalized_abs_{side}_{safe_pname}.png",
                    )
                    plt.close("all")
                    print(f"  + fluent_comparison/bl_fluent_normalized_abs_{side}_{safe_pname}.png")

    if args.show_plots:
        plt.show()

    print("\nBoundary layer analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
