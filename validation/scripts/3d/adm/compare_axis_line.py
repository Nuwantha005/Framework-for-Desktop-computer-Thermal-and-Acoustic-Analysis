#!/usr/bin/env python3
"""Compare direct CFD axis-line data against the 3D panel+ADM solution."""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    HEADER_ALIASES,
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

from core.io.case_loader import CaseLoader


def load_direct_csv_pandas(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load Fluent CSV file using pandas for speed and return points and fields."""
    df = pd.read_csv(path, skipinitialspace=True)
    
    def get_column(aliases):
        for col in df.columns:
            col_norm = col.strip().lower().replace("-", "").replace("_", "")
            for alias in aliases:
                alias_norm = alias.strip().lower().replace("-", "").replace("_", "")
                if col_norm == alias_norm:
                    return df[col].values
        # Substring fallback
        for col in df.columns:
            col_norm = col.strip().lower().replace("-", "").replace("_", "")
            for alias in aliases:
                alias_norm = alias.strip().lower().replace("-", "").replace("_", "")
                if len(alias_norm) >= 4 and alias_norm in col_norm:
                    return df[col].values
        return None

    x = get_column(HEADER_ALIASES["x"])
    y = get_column(HEADER_ALIASES["y"])
    z = get_column(HEADER_ALIASES["z"])
    if x is None or y is None or z is None:
        raise ValueError(f"Could not identify x/y/z coordinates in {path}")
        
    points = np.column_stack([x, y, z]).astype(np.float64)
    
    fields = {}
    for quantity in ("pressure", "temperature", "x_velocity", "y_velocity", "z_velocity"):
        val = get_column(HEADER_ALIASES[quantity])
        if val is not None:
            fields[quantity] = val.astype(np.float64)
            
    speed = get_column(HEADER_ALIASES["velocity_magnitude"])
    if speed is not None:
        fields["velocity_magnitude"] = speed.astype(np.float64)
    elif {"x_velocity", "y_velocity", "z_velocity"} <= set(fields):
        vel = np.column_stack([fields["x_velocity"], fields["y_velocity"], fields["z_velocity"]])
        fields["velocity_magnitude"] = np.linalg.norm(vel, axis=1)
        
    return points, fields


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
    
    # Resolve reference file path
    line_path = None
    if line_data_path is not None:
        line_path = Path(line_data_path)
    else:
        # Search in default location
        export_dir = case_dir / "fluent_case" / "export" / "panel"
        if export_dir.exists():
            for name in ["line_data", "field_data", "FFF 1.9-Setup-Output"]:
                if (export_dir / name).exists():
                    line_path = export_dir / name
                    break
            if line_path is None:
                # Fallback to first file in directory
                files = [f for f in export_dir.iterdir() if f.is_file()]
                if files:
                    line_path = files[0]
                    
    if line_path is None or not line_path.exists():
        raise FileNotFoundError(f"CFD reference data not found in case: {case_dir}")

    print(f"Loading reference CFD data from: {line_path}")
    points, fields = load_direct_csv_pandas(line_path)

    # Check if dataset is 1D (pre-extracted line) or 3D (volume point cloud)
    spans = np.ptp(points, axis=0)
    tol = max(np.max(spans), 1.0) * 1e-3
    varying = np.flatnonzero(spans > tol)

    out_dir = output_dir or default_output_dir(case_dir, "axis_line")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_all: dict[str, object] = {
        "case_dir": str(case_dir),
        "reference_source": str(line_path.resolve()),
        "runs": {}
    }

    if len(varying) == 1:
        # Backward compatibility: Single 1D line dataset
        print("Detected 1D axis-line reference dataset.")
        line_axis, fixed_axes = detect_line(points)
        axis_index = {"x": 0, "y": 1, "z": 2}[line_axis]
        axis_coordinate = points[:, axis_index]
        order = np.argsort(axis_coordinate)
        axis_coordinate = axis_coordinate[order]

        panel = sample_panel_fields(
            case_dir=case_dir,
            points=points,
            quantities=requested,
            mesh_level=mesh_level,
            volume_field_path=volume_field_path,
        )

        csv_columns: dict[str, np.ndarray] = {line_axis: axis_coordinate}
        run_summary = {
            "panel_source": panel.source,
            "line": {
                "line_axis": line_axis,
                "fixed_axes": [{name: value} for name, value in fixed_axes],
            },
            "warnings": panel.warnings.copy(),
            "quantities": {},
            "skipped_quantities": {},
        }

        fixed_title = ", ".join(f"{name.upper()} = {value:.6f} m" for name, value in fixed_axes)
        for quantity in requested:
            reference = fields.get(quantity)
            test = panel.fields.get(quantity)
            if reference is None:
                run_summary["skipped_quantities"][quantity] = "Reference data does not contain this quantity."
                continue
            reference = np.asarray(reference)[order]
            csv_columns[f"fluent_{quantity}"] = reference
            if test is None or np.all(np.isnan(test)):
                run_summary["skipped_quantities"][quantity] = "Panel-side data does not contain this quantity."
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
            run_summary["quantities"][quantity] = {
                "metrics": metrics_to_dict(metrics),
                "reference_min": float(np.nanmin(reference)),
                "reference_max": float(np.nanmax(reference)),
                "panel_min": float(np.nanmin(test)),
                "panel_max": float(np.nanmax(test)),
            }
            print(f"[axis-line] {quantity}: {format_metrics(metrics)}")

        write_csv(out_dir / "axis_line_samples.csv", csv_columns)
        write_json(out_dir / "axis_line_metrics.json", run_summary)
        summary_all["runs"]["default"] = run_summary
        return run_summary

    else:
        # 3D dataset: Extract lines for all configured actuator disks
        print("Detected 3D volume point cloud reference dataset.")
        case = CaseLoader.load_case(case_dir, mesh_level_index=mesh_level)
        disks = case.config.actuator_disks
        if not disks:
            raise ValueError("No actuator disks found in case config to extract axis lines from 3D reference data.")

        from scipy.spatial import KDTree

        print(f"Building KDTree for {len(points)} Fluent points...")
        tree = KDTree(points)

        default_summary = None

        for disk in disks:
            print(f"\nProcessing axis-line for actuator disk: '{disk.name}'")
            center = np.asarray(disk.center, dtype=np.float64)
            normal = np.asarray(disk.normal, dtype=np.float64)
            normal = normal / np.linalg.norm(normal)

            # Project Fluent points onto normal vector
            rel = points - center
            t = np.dot(rel, normal)
            t_min, t_max = np.min(t), np.max(t)

            # Define plotting axis coordinate
            main_axis_index = int(np.argmax(np.abs(normal)))
            line_axis = ["x", "y", "z"][main_axis_index]

            N_samples = 200
            t_line = np.linspace(t_min, t_max, N_samples)
            points_line = center + np.outer(t_line, normal)
            axis_coordinate = points_line[:, main_axis_index]
            
            # Query KDTree
            distances, indices = tree.query(points_line, k=4)
            weights = 1.0 / np.maximum(distances, 1e-12)
            weights /= np.sum(weights, axis=1, keepdims=True)

            interpolated_fields = {}
            for name, vals in fields.items():
                interpolated_fields[name] = np.sum(vals[indices] * weights, axis=1)

            # Sample panel-side solver results
            panel = sample_panel_fields(
                case_dir=case_dir,
                points=points_line,
                quantities=requested,
                mesh_level=mesh_level,
                volume_field_path=volume_field_path,
            )

            csv_columns = {line_axis: axis_coordinate}
            fixed_axes = []
            for idx, name in enumerate(["x", "y", "z"]):
                if idx != main_axis_index:
                    fixed_axes.append((name, float(center[idx])))

            run_summary = {
                "panel_source": panel.source,
                "line": {
                    "line_axis": line_axis,
                    "fixed_axes": [{name: value} for name, value in fixed_axes],
                },
                "warnings": panel.warnings.copy(),
                "quantities": {},
                "skipped_quantities": {},
            }

            fixed_title = f"Disk: {disk.name}, " + ", ".join(f"{name.upper()} = {value:.6f} m" for name, value in fixed_axes)

            for quantity in requested:
                reference = interpolated_fields.get(quantity)
                test = panel.fields.get(quantity)
                if reference is None:
                    run_summary["skipped_quantities"][quantity] = "Reference data does not contain this quantity."
                    continue
                csv_columns[f"fluent_{quantity}"] = reference
                if test is None or np.all(np.isnan(test)):
                    run_summary["skipped_quantities"][quantity] = "Panel-side data does not contain this quantity."
                    continue
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
                    output_path=out_dir / f"axis_line_{disk.name}_{safe_quantity_name(quantity)}.png",
                    show_plots=show_plots,
                )
                run_summary["quantities"][quantity] = {
                    "metrics": metrics_to_dict(metrics),
                    "reference_min": float(np.nanmin(reference)),
                    "reference_max": float(np.nanmax(reference)),
                    "panel_min": float(np.nanmin(test)),
                    "panel_max": float(np.nanmax(test)),
                }
                print(f"  {quantity}: {format_metrics(metrics)}")

            write_csv(out_dir / f"axis_line_{disk.name}_samples.csv", csv_columns)
            write_json(out_dir / f"axis_line_{disk.name}_metrics.json", run_summary)
            summary_all["runs"][disk.name] = run_summary
            if default_summary is None:
                default_summary = run_summary

        write_json(out_dir / "axis_line_summary_all.json", summary_all)
        return default_summary or {}


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

    # Font sizes
    label_fs = 14
    legend_fs = 13
    tick_fs = 12

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
    axes[0].set_ylabel(quantity_label(quantity), fontsize=label_fs)
    # Title removed per request (fixed axes info omitted)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=legend_fs)
    axes[0].tick_params(axis="both", labelsize=tick_fs)

    axes[1].plot(axis_coordinate, diff, color="tab:red", linewidth=1.8)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    axes[1].set_xlabel(f"{line_axis.upper()} [m]", fontsize=label_fs)
    axes[1].set_ylabel("Error", fontsize=label_fs)
    axes[1].grid(alpha=0.25)
    axes[1].tick_params(axis="both", labelsize=tick_fs)

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
