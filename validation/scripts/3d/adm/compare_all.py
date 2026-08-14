#!/usr/bin/env python3
"""Run all direct-data ADM validation comparisons for a case."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

from common import canonicalize_quantities, default_output_dir, write_json
from compare_axis_line import run_axis_line_comparison
from compare_cut_plane import run_cut_plane_comparison


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run all ADM direct-data comparisons")
    parser.add_argument("case_dir", type=Path, help="Path to case directory")
    parser.add_argument("--mesh-level", type=int, default=0, help="Mesh refinement level index")
    parser.add_argument(
        "--quantities",
        nargs="+",
        default=["velocity_magnitude", "temperature"],
        help="Quantities to compare",
    )
    parser.add_argument(
        "--cut-data",
        type=Path,
        default=None,
        help="Override path to cut-plane direct CSV export",
    )
    parser.add_argument(
        "--line-data",
        type=Path,
        default=None,
        help="Override path to axis-line direct CSV export",
    )
    parser.add_argument(
        "--volume-field",
        type=Path,
        default=None,
        help="Override path to saved panel volume field (.vts)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root (default: <case>/out/validation/adm)",
    )
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    if not args.show_plots:
        matplotlib.use("Agg")

    case_dir = args.case_dir.resolve()
    output_root = args.output_dir or (case_dir / "out" / "validation" / "adm")
    output_root.mkdir(parents=True, exist_ok=True)
    quantities = canonicalize_quantities(args.quantities)

    cut_summary = run_cut_plane_comparison(
        case_dir=case_dir,
        mesh_level=args.mesh_level,
        quantities=quantities,
        cut_data_path=args.cut_data,
        volume_field_path=args.volume_field,
        output_dir=output_root / "cut_plane",
        show_plots=args.show_plots,
    )
    line_summary = run_axis_line_comparison(
        case_dir=case_dir,
        mesh_level=args.mesh_level,
        quantities=quantities,
        line_data_path=args.line_data,
        volume_field_path=args.volume_field,
        output_dir=output_root / "axis_line",
        show_plots=args.show_plots,
    )

    write_json(
        output_root / "summary.json",
        {
            "case_dir": str(case_dir),
            "quantities": quantities,
            "cut_plane": cut_summary,
            "axis_line": line_summary,
        },
    )
    print(f"[adm-validation] Wrote combined summary to {output_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
