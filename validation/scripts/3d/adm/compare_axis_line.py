#!/usr/bin/env python3
"""Compare direct CFD axis-line data against the 3D panel+ADM solution."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from common import (
    canonicalize_quantities,
    compare_quantity,
    default_output_dir,
    detect_line,
    format_metrics,
    load_direct_csv,
    metrics_to_dict,
    quantity_label,
    quantity_meta,
    safe_quantity_name,
    sample_panel_fields,
    write_csv,
    write_json,
)


def run_axis_line_comparison(
    case_dir: Path,
    mesh_level: int = 0,
    quantities: list[str] | None = None,
    line_data_path: Path | None = None,
    volume_field_path: Path | None = None,
    output_dir: Path | None = None,
    show_plots: bool = False,
) -> dict[str, object]:
    """Run axis-line comparison and save plots/metrics."""
    case_dir = Path(case_dir).resolve()
    requested = canonicalize_quantities(quantities or ["velocity_magnitude", "temperature"])
    line_path = line_data_path or (case_dir / "fluent_case" / "export" / "panel" / "line_data")
    direct = load_direct_csv(line_path)
    line_axis, fixed_axes = detect_line(direct.points)

    panel = sample_panel_fields(
        case_dir=case_dir,
        points=direct.points,
        quantities=requested,
        mesh_level=mesh_level,
        volume_field_path=volume_field_path,
    )

    out_dir = output_dir or default_output_dir(case_dir, "axis_line")
    out_dir.mkdir(parents=True, exist_ok=True)

    axis_index = {"x": 0, "y": 1, "z": 2}[line_axis]
    axis_coordinate = direct.points[:, axis_index]
    order = np.argsort(axis_coordinate)
    axis_coordinate = axis_coordinate[order]

    csv_columns: dict[str, np.ndarray] = {line_axis: axis_coordinate}
    summary: dict[str, object] = {
        "case_dir": str(case_dir),
        "reference_line_data": str(line_path.resolve()),
        "panel_source": panel.source,
        "line": {
            "line_axis": line_axis,
            "fixed_axes": [{name: value} for name, value in fixed_axes],
        },
        "warnings": panel.warnings.copy(),
        "quantities": {},
        "skipped_quantities": {},
    }

    for warning in panel.warnings:
        print(f"[axis-line warning] {warning}")

    fixed_title = ", ".join(f"{name.upper()} = {value:.6f} m" for name, value in fixed_axes)
    comparable = 0
    for quantity in requested:
        reference = direct.fields.get(quantity)
        test = panel.fields.get(quantity)
        if reference is None:
            message = f"Reference axis-line export does not contain '{quantity}'."
            summary["skipped_quantities"][quantity] = message
            print(f"[axis-line skip] {message}")
            continue

        reference = np.asarray(reference)[order]
        csv_columns[f"fluent_{quantity}"] = reference
        if test is None or np.all(np.isnan(test)):
            message = f"Panel-side data does not contain '{quantity}'."
            summary["skipped_quantities"][quantity] = message
            print(f"[axis-line skip] {message}")
            csv_columns[f"panel_{quantity}"] = np.full_like(reference, np.nan)
            csv_columns[f"error_{quantity}"] = np.full_like(reference, np.nan)
            continue

        test = np.asarray(test)[order]
        diff = test - reference
        metrics = compare_quantity(reference, test)

        csv_columns[f"panel_{quantity}"] = test
        csv_columns[f"error_{quantity}"] = diff
        _plot_axis_line(
            axis_coordinate=axis_coordinate,
            quantity=quantity,
            reference=reference,
            test=test,
            line_axis=line_axis,
            fixed_title=fixed_title,
            output_path=out_dir / f"axis_line_{safe_quantity_name(quantity)}.png",
            show_plots=show_plots,
        )

        comparable += 1
        summary["quantities"][quantity] = {
            "metrics": metrics_to_dict(metrics),
            "reference_min": float(np.nanmin(reference)),
            "reference_max": float(np.nanmax(reference)),
            "panel_min": float(np.nanmin(test)),
            "panel_max": float(np.nanmax(test)),
            "difference_min": float(np.nanmin(diff)),
            "difference_max": float(np.nanmax(diff)),
        }
        print(f"[axis-line] {quantity}: {format_metrics(metrics)}")

    write_csv(out_dir / "axis_line_samples.csv", csv_columns)
    write_json(out_dir / "axis_line_metrics.json", summary)

    if comparable == 0:
        print("[axis-line] No comparable quantities were available.")

    if show_plots:
        plt.show()
    else:
        plt.close("all")

    return summary


def _plot_axis_line(
    axis_coordinate: np.ndarray,
    quantity: str,
    reference: np.ndarray,
    test: np.ndarray,
    line_axis: str,
    fixed_title: str,
    output_path: Path,
    show_plots: bool,
) -> None:
    """Plot reference vs panel values and their difference along the axis."""
    diff = test - reference
    meta = quantity_meta(quantity)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )

    axes[0].plot(axis_coordinate, reference, label="Fluent", linewidth=2.0)
    axes[0].plot(axis_coordinate, test, label="Panel + ADM", linewidth=2.0, linestyle="--")
    axes[0].set_ylabel(quantity_label(quantity))
    axes[0].set_title(f"Axis-Line Comparison: {meta['label']}\n{fixed_title}")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(axis_coordinate, diff, color="tab:red", linewidth=1.8)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    axes[1].set_xlabel(f"{line_axis.upper()} [m]")
    axes[1].set_ylabel("Error")
    axes[1].grid(alpha=0.25)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if not show_plots:
        plt.close(fig)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="ADM axis-line comparison against direct CFD export")
    parser.add_argument("case_dir", type=Path, help="Path to case directory")
    parser.add_argument("--mesh-level", type=int, default=0, help="Mesh refinement level index")
    parser.add_argument(
        "--quantities",
        nargs="+",
        default=["velocity_magnitude", "temperature"],
        help="Quantities to compare",
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
        help="Output directory (default: <case>/out/validation/adm/axis_line)",
    )
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    if not args.show_plots:
        matplotlib.use("Agg")

    run_axis_line_comparison(
        case_dir=args.case_dir,
        mesh_level=args.mesh_level,
        quantities=args.quantities,
        line_data_path=args.line_data,
        volume_field_path=args.volume_field,
        output_dir=args.output_dir,
        show_plots=args.show_plots,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
