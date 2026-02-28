#!/usr/bin/env python3
"""
Solver Comparison Demo

Compare multiple panel method formulations on the same case,
optionally including OpenFOAM CFD reference data. Generates Vt and Cp
envelope overlay plots, arc-length line charts, difference plots,
error metric tables, and solver ranking charts.

Usage:
    # Panel method solvers only
    python demos/demo_solver_comparison.py cases/rounded_square
    python demos/demo_solver_comparison.py cases/cylinder_flow --show
    python demos/demo_solver_comparison.py cases/rounded_square --solvers constant linear vortex --mesh-level 3

    # With OpenFOAM reference
    python demos/demo_solver_comparison.py cases/rounded_square --of-case of_case/cases/level_4
    python demos/demo_solver_comparison.py cases/single_square --of-case of_case/cases/level_4 --mesh-level -1

    # All available solvers vs OpenFOAM
    python demos/demo_solver_comparison.py cases/rounded_square --solvers constant linear --of-case of_case/cases/level_4 --show

Available solver short names:
    constant   — constant-strength source panels
    linear     — linear-strength source panels
    vortex     — linear-strength vortex panels
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Add project root to path (for validation imports)
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.io import CaseLoader
from solvers.comparison import SolverComparisonRunner
from visualization.solver_comparison import SolverComparisonVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="Compare panel method solvers on the same case",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "case_dir",
        type=Path,
        help="Path to case directory (must contain case.yaml)",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["constant", "linear", "vortex"],
        help="Solver types to compare (default: constant linear vortex)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Custom labels for each solver (must match --solvers count)",
    )
    parser.add_argument(
        "--of-case",
        type=Path,
        default=None,
        help="Path to OpenFOAM case (relative to case_dir or absolute). "
             "e.g. of_case/cases/level_4",
    )
    parser.add_argument(
        "--of-label",
        type=str,
        default="OpenFOAM",
        help="Label for OpenFOAM reference in plots (default: OpenFOAM)",
    )
    parser.add_argument(
        "--mesh-level",
        type=int,
        default=-1,
        help="Mesh refinement level index (-1 = finest, default: -1)",
    )
    parser.add_argument(
        "--envelope-scale",
        type=float,
        default=0.3,
        help="Scale factor for envelope displacement (default: 0.3)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output base directory (default: <case_dir>/out). "
             "Plots are saved under <output>/solver_comparison/",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save plots to disk",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved images (default: 150)",
    )
    args = parser.parse_args()

    # Validate case directory
    case_dir = args.case_dir.resolve()
    if not case_dir.exists():
        print(f"Error: case directory not found: {case_dir}")
        return 1

    # Resolve OpenFOAM case path (relative to case_dir or absolute)
    of_case_dir = None
    if args.of_case is not None:
        if args.of_case.is_absolute():
            of_case_dir = args.of_case
        else:
            of_case_dir = (case_dir / args.of_case).resolve()
        if not of_case_dir.exists():
            print(f"Error: OpenFOAM case not found: {of_case_dir}")
            return 1

    # Load case
    print(f"Loading case: {case_dir}")
    level_label = "finest" if args.mesh_level == -1 else f"level {args.mesh_level}"
    print(f"  Mesh level: {args.mesh_level} ({level_label})")

    case = CaseLoader.load_case(case_dir, mesh_level_index=args.mesh_level)
    print(f"  {case.name}: {case.num_panels} panels, "
          f"V∞={case.v_inf}, AoA={case.aoa}°")

    if of_case_dir is not None:
        print(f"  OpenFOAM reference: {of_case_dir}")

    # Run comparison
    runner = SolverComparisonRunner(case)
    result = runner.run(
        solver_types=args.solvers,
        labels=args.labels,
        of_case_dir=of_case_dir,
        of_label=args.of_label,
        verbose=True,
    )

    # Visualize — output goes to <case_dir>/out/solver_comparison/
    output_dir = args.output or case.output_dir
    save = not args.no_save

    viz = SolverComparisonVisualizer(result, output_dir=output_dir)
    viz.plot_all(
        show=args.show,
        save=save,
        dpi=args.dpi,
        envelope_scale=args.envelope_scale,
    )

    # Print best solver if ranking is available
    if result.ranking:
        best_label, best_pct = result.ranking[0]
        print(f"\nBest solver: {best_label} ({best_pct:.2f}% relative Vt error)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
