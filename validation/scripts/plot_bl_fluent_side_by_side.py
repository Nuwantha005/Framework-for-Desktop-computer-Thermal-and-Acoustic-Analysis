#!/usr/bin/env python3
"""Generate boundary-layer Fluent side-by-side comparison plots.

This script focuses on absolute side-by-side validation views:
- Envelope (BL vs Fluent)
- Contour (BL vs Fluent)
- Normalized contour (BL vs Fluent)
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
from visualization.bl_plots import (
    plot_bl_fluent_contour_normalized_side_by_side,
    plot_bl_fluent_contour_side_by_side,
    plot_bl_fluent_envelope_side_by_side,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate BL Fluent side-by-side plots")
    parser.add_argument("case_dir", type=Path, help="Path to case directory")
    parser.add_argument("--solver-type", type=str, default=None, help="Panel solver type")
    parser.add_argument("--mesh-level", type=int, default=-1, help="Mesh refinement level index")
    parser.add_argument("--profiles", nargs="+", default=None, help="BL profiles")
    parser.add_argument("--transition", type=str, default=None, choices=["michel", "en"], help="Transition model")
    parser.add_argument("--nu", type=float, default=None, help="Kinematic viscosity override [m^2/s]")
    parser.add_argument("--compare-profile", type=str, default=None, help="Single profile name for comparison")
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <case>/out/boundary_layer/fluent_comparison/side_by_side)",
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
    bl = runner.run(profiles=profiles, nu=args.nu, transition_model=transition, reconstruct=True)

    output_dir = args.output_dir or (case.output_dir / "boundary_layer" / "fluent_comparison" / "side_by_side")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {output_dir}")

    compare_profiles = [args.compare_profile] if args.compare_profile else list(bl.upper.fields.keys())
    comp_runner = BLComparisonRunner(case, bl)

    for profile_name in compare_profiles:
        comp_result = comp_runner.run(profile_name=profile_name)
        if not comp_result.has_fluent_data:
            print("Warning: Fluent data not available at fluent_case/export/viscous_bl/")
            break

        pname = comp_result.profile_name
        safe = pname.lower().replace(" ", "_").replace("-", "_")

        plot_bl_fluent_envelope_side_by_side(
            bl,
            comp_result,
            profile_name=pname,
            title=f"Absolute velocity envelope - {pname} vs Fluent - {case.name}",
            output_path=output_dir / f"bl_fluent_envelope_abs_{safe}.png",
        )
        plt.close("all")

        for side in ["upper", "lower"]:
            field = bl.sides[side].fields.get(pname)
            fluent_field = comp_result.sides[side]
            if field is None or fluent_field is None:
                continue

            plot_bl_fluent_contour_side_by_side(
                field,
                fluent_field,
                title=f"{side.capitalize()} absolute velocity - {pname} vs Fluent - {case.name}",
                output_path=output_dir / f"bl_fluent_contour_abs_{side}_{safe}.png",
            )
            plt.close("all")

            plot_bl_fluent_contour_normalized_side_by_side(
                field,
                fluent_field,
                title=f"{side.capitalize()} normalized velocity - {pname} vs Fluent - {case.name}",
                output_path=output_dir / f"bl_fluent_normalized_abs_{side}_{safe}.png",
            )
            plt.close("all")

        print(f"  + Generated side-by-side plots for profile: {pname}")

    if args.show_plots:
        plt.show()

    print("Side-by-side comparison plotting complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
