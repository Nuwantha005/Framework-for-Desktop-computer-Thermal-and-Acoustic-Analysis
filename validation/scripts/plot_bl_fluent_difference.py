#!/usr/bin/env python3
"""Generate boundary-layer Fluent difference comparison plots.

This script focuses only on difference-style validation outputs:
- Velocity difference contours
- Normalized difference contours
- Difference envelope plots
- Wall-quantity comparison and metrics report
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
    plot_bl_comparison_report,
    plot_bl_fluent_comparison,
    plot_bl_fluent_comparison_two_sides,
    plot_bl_velocity_contour_normalized_comparison,
    plot_bl_velocity_envelope_comparison,
    plot_bl_wall_comparison,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate BL Fluent difference plots")
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
        help="Output directory (default: <case>/out/boundary_layer/fluent_comparison/difference)",
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

    output_dir = args.output_dir or (case.output_dir / "boundary_layer" / "fluent_comparison" / "difference")
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

        plot_bl_fluent_comparison_two_sides(
            bl,
            comp_result,
            profile_name=pname,
            title=f"Velocity difference - {pname} - {case.name}",
            output_path=output_dir / f"bl_fluent_velocity_diff_{safe}.png",
        )
        plt.close("all")

        plot_bl_wall_comparison(
            bl,
            comp_result.fluent_result,
            quantities=["Ue", "Cf", "delta"],
            profile_name=pname,
            title=f"Wall quantities - {pname} - {case.name}",
            output_path=output_dir / f"bl_fluent_wall_{safe}.png",
        )
        plt.close("all")

        plot_bl_velocity_envelope_comparison(
            bl,
            comp_result,
            profile_name=pname,
            title=f"Velocity difference envelope - {pname} vs Fluent - {case.name}",
            output_path=output_dir / f"bl_fluent_envelope_diff_{safe}.png",
        )
        plt.close("all")

        for side in ["upper", "lower"]:
            field = bl.sides[side].fields.get(pname)
            fluent_field = comp_result.sides[side]
            if field is None or fluent_field is None:
                continue

            plot_bl_fluent_comparison(
                field,
                fluent_field,
                title=f"{side.capitalize()} - {pname} vs Fluent - {case.name}",
                output_path=output_dir / f"bl_fluent_diff_{side}_{safe}.png",
            )
            plt.close("all")

            plot_bl_velocity_contour_normalized_comparison(
                field,
                fluent_field,
                title=f"{side.capitalize()} normalized difference - {pname} - {case.name}",
                output_path=output_dir / f"bl_fluent_normalized_diff_{side}_{safe}.png",
            )
            plt.close("all")

        plot_bl_comparison_report(
            comp_result,
            title=f"Comparison report - {pname} - {case.name}",
            output_path=output_dir / f"bl_fluent_report_{safe}.png",
        )
        plt.close("all")

        print(f"  + Generated difference plots for profile: {pname}")

    if args.show_plots:
        plt.show()

    print("Difference comparison plotting complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
