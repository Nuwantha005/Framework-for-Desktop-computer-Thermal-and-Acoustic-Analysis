#!/usr/bin/env python3
"""Generate wall quantity envelope plots comparing BL solver vs Fluent.

This script produces envelope plots for wall quantities (Ue, Cf, delta, Cp)
wrapped around the body geometry. Two plot variants are available:

- Side-by-side: BL solver on left panel, Fluent on right panel
- Overlay: Both results on the same body for direct comparison

Usage:
    python plot_bl_fluent_wall_envelopes.py cases/rounded_rectangle
    python plot_bl_fluent_wall_envelopes.py cases/rounded_rectangle --mode overlay
    python plot_bl_fluent_wall_envelopes.py cases/rounded_rectangle --quantities Ue Cf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.io.case_loader import CaseLoader
from solvers.boundary_layer.runner import BoundaryLayerRunner
from validation.adapters.fluent import BLComparisonRunner
from visualization.bl_wall_envelope_plots import (
    plot_wall_quantity_envelope_overlay,
    plot_wall_quantity_envelope_side_by_side,
    plot_wall_quantity_envelopes_grid,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate wall quantity envelope plots (BL Solver vs Fluent)"
    )
    parser.add_argument("case_dir", type=Path, help="Path to case directory")
    parser.add_argument(
        "--solver-type", type=str, default=None, help="Panel solver type"
    )
    parser.add_argument(
        "--mesh-level", type=int, default=-1, help="Mesh refinement level index"
    )
    parser.add_argument(
        "--profiles", nargs="+", default=None, help="BL profiles to solve"
    )
    parser.add_argument(
        "--transition",
        type=str,
        default=None,
        choices=["michel", "en"],
        help="Transition model",
    )
    parser.add_argument(
        "--nu", type=float, default=None, help="Kinematic viscosity override [m^2/s]"
    )
    parser.add_argument(
        "--compare-profile",
        type=str,
        default=None,
        help="Single profile name for comparison (default: first available)",
    )
    parser.add_argument(
        "--quantities",
        nargs="+",
        default=["Ue", "Cf", "delta", "Cp"],
        help="Wall quantities to plot (default: Ue Cf delta Cp)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["side_by_side", "overlay", "grid", "both"],
        help="Plot mode: side_by_side, overlay, grid, or both (default: both)",
    )
    parser.add_argument(
        "--scale", type=float, default=0.15, help="Envelope displacement scale factor"
    )
    parser.add_argument(
        "--show-plots", action="store_true", help="Display plots interactively"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <case>/out/boundary_layer/fluent_comparison/wall_envelopes)",
    )
    args = parser.parse_args()

    if not args.show_plots:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case_dir = args.case_dir.resolve()
    if not case_dir.exists():
        print(f"Error: Case directory not found: {case_dir}")
        return 1

    print(f"Loading case: {case_dir}")
    case = CaseLoader.load_case(case_dir, mesh_level_index=args.mesh_level)
    solver = case.create_solver(solver_type=args.solver_type)
    solver.solve()

    runner = BoundaryLayerRunner(case, solver)
    bl_cfg = case.config.boundary_layer
    profiles = args.profiles or list(bl_cfg.profiles)
    transition = args.transition or bl_cfg.transition_model
    bl = runner.run(
        profiles=profiles, nu=args.nu, transition_model=transition, reconstruct=True
    )

    output_dir = args.output_dir or (
        case.output_dir / "boundary_layer" / "fluent_comparison" / "wall_envelopes"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {output_dir}")

    compare_profile = args.compare_profile or (
        bl.profile_names[0] if bl.profile_names else None
    )
    comp_runner = BLComparisonRunner(case, bl)
    comp_result = comp_runner.run(profile_name=compare_profile)

    if not comp_result.has_fluent_data:
        print("Warning: Fluent data not available at fluent_case/export/viscous_bl/")
        print("Plots will show BL solver data only.")

    pname = comp_result.profile_name or "unknown"
    safe = pname.lower().replace(" ", "_").replace("-", "_")

    # Generate plots based on mode
    if args.mode in ("side_by_side", "both"):
        for qty in args.quantities:
            qty_safe = qty.lower()
            plot_wall_quantity_envelope_side_by_side(
                bl,
                comp_result,
                quantity=qty,
                profile_name=pname,
                scale=args.scale,
                title=f"{qty} envelope — {pname} vs Fluent — {case.name}",
                output_path=output_dir / f"wall_{qty_safe}_side_by_side_{safe}.png",
            )
            plt.close("all")
        print(f"  + Generated side-by-side envelope plots for: {', '.join(args.quantities)}")

    if args.mode in ("overlay", "both"):
        for qty in args.quantities:
            qty_safe = qty.lower()
            plot_wall_quantity_envelope_overlay(
                bl,
                comp_result,
                quantity=qty,
                profile_name=pname,
                scale=args.scale,
                title=f"{qty} envelope — {pname} vs Fluent — {case.name}",
                output_path=output_dir / f"wall_{qty_safe}_overlay_{safe}.png",
                show_difference=True,
            )
            plt.close("all")
        print(f"  + Generated overlay envelope plots for: {', '.join(args.quantities)}")

    if args.mode == "grid":
        plot_wall_quantity_envelopes_grid(
            bl,
            comp_result,
            quantities=args.quantities,
            profile_name=pname,
            scale=args.scale,
            mode="overlay",
            title=f"Wall quantities — {pname} vs Fluent — {case.name}",
            output_path=output_dir / f"wall_quantities_grid_{safe}.png",
        )
        plt.close("all")
        print("  + Generated grid envelope plot")

    if args.show_plots:
        plt.show()

    print("Wall envelope plotting complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
