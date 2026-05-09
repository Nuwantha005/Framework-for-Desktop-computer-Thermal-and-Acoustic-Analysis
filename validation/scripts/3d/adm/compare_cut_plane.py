#!/usr/bin/env python3
"""Compare a direct CFD cut plane against the 3D panel+ADM solution."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib import cm, colors

from common import (
    DirectCSVData,
    canonicalize_quantities,
    compare_quantity,
    default_output_dir,
    detect_plane,
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


def run_cut_plane_comparison(
    case_dir: Path,
    mesh_level: int = 0,
    quantities: list[str] | None = None,
    cut_data_path: Path | None = None,
    volume_field_path: Path | None = None,
    output_dir: Path | None = None,
    show_plots: bool = False,
) -> dict[str, object]:
    """Run cut-plane comparison and save plots/metrics."""
    case_dir = Path(case_dir).resolve()
    requested = canonicalize_quantities(quantities or ["velocity_magnitude", "temperature"])
    cut_path = cut_data_path or (case_dir / "fluent_case" / "export" / "panel" / "cut_data")
    direct = load_direct_csv(cut_path)
    normal_axis, plane_axes, plane_value = detect_plane(direct.points)

    panel = sample_panel_fields(
        case_dir=case_dir,
        points=direct.points,
        quantities=requested,
        mesh_level=mesh_level,
        volume_field_path=volume_field_path,
    )

    out_dir = output_dir or default_output_dir(case_dir, "cut_plane")
    out_dir.mkdir(parents=True, exist_ok=True)

    x_idx = {"x": 0, "y": 1, "z": 2}[plane_axes[0]]
    y_idx = {"x": 0, "y": 1, "z": 2}[plane_axes[1]]
    x = direct.points[:, x_idx]
    y = direct.points[:, y_idx]

    csv_columns: dict[str, np.ndarray] = {
        plane_axes[0]: x,
        plane_axes[1]: y,
    }
    summary: dict[str, object] = {
        "case_dir": str(case_dir),
        "reference_cut_data": str(cut_path.resolve()),
        "panel_source": panel.source,
        "plane": {
            "normal_axis": normal_axis,
            "plane_value": plane_value,
            "plane_axes": list(plane_axes),
        },
        "warnings": panel.warnings.copy(),
        "quantities": {},
        "skipped_quantities": {},
    }

    for warning in panel.warnings:
        print(f"[cut-plane warning] {warning}")

    comparable = 0
    for quantity in requested:
        reference = direct.fields.get(quantity)
        test = panel.fields.get(quantity)
        if reference is None:
            message = f"Reference cut-plane export does not contain '{quantity}'."
            summary["skipped_quantities"][quantity] = message
            print(f"[cut-plane skip] {message}")
            continue
        if test is None or np.all(np.isnan(test)):
            message = f"Panel-side data does not contain '{quantity}'."
            summary["skipped_quantities"][quantity] = message
            print(f"[cut-plane skip] {message}")
            csv_columns[f"panel_{quantity}"] = np.full_like(reference, np.nan)
            csv_columns[f"error_{quantity}"] = np.full_like(reference, np.nan)
            csv_columns[f"fluent_{quantity}"] = reference
            continue

        metrics = compare_quantity(reference, test)
        diff = np.asarray(test) - np.asarray(reference)
        csv_columns[f"fluent_{quantity}"] = np.asarray(reference)
        csv_columns[f"panel_{quantity}"] = np.asarray(test)
        csv_columns[f"error_{quantity}"] = diff

        plot_x, plot_y, plot_axes = _prepare_plot_axes(plane_axes, x, y)

        _plot_absolute_stack(
            direct=direct,
            x=plot_x,
            y=plot_y,
            quantity=quantity,
            plane_axes=plot_axes,
            plane_title=f"{normal_axis.upper()} = {plane_value:.6f} m",
            reference=reference,
            test=test,
            output_path=out_dir / f"cut_plane_{safe_quantity_name(quantity)}_stack.png",
            show_plots=show_plots,
        )
        _plot_difference(
            x=plot_x,
            y=plot_y,
            quantity=quantity,
            plane_axes=plot_axes,
            plane_title=f"{normal_axis.upper()} = {plane_value:.6f} m",
            difference=diff,
            output_path=out_dir / f"cut_plane_{safe_quantity_name(quantity)}_difference.png",
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
        print(f"[cut-plane] {quantity}: {format_metrics(metrics)}")

    write_csv(out_dir / "cut_plane_samples.csv", csv_columns)
    write_json(out_dir / "cut_plane_metrics.json", summary)

    if comparable == 0:
        print("[cut-plane] No comparable quantities were available.")

    if show_plots:
        plt.show()
    else:
        plt.close("all")

    return summary


def _prepare_plot_axes(
    plane_axes: tuple[str, str],
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, str]]:
    """Reorder in-plane axes for more readable plotting.

    When the section includes ``z``, plot it horizontally.
    """
    if "z" in plane_axes and plane_axes[0] != "z":
        return y, x, (plane_axes[1], plane_axes[0])
    return x, y, plane_axes


def _plot_absolute_stack(
    direct: DirectCSVData,
    x: np.ndarray,
    y: np.ndarray,
    quantity: str,
    plane_axes: tuple[str, str],
    plane_title: str,
    reference: np.ndarray,
    test: np.ndarray,
    output_path: Path,
    show_plots: bool,
) -> None:
    """Plot stacked absolute fields for reference and panel results."""
    valid = ~(np.isnan(reference) | np.isnan(test))
    if np.count_nonzero(valid) < 3:
        return

    meta = quantity_meta(quantity)
    triangulation = mtri.Triangulation(x[valid], y[valid])
    vmin = float(min(np.nanmin(reference[valid]), np.nanmin(test[valid])))
    vmax = float(max(np.nanmax(reference[valid]), np.nanmax(test[valid])))
    if np.isclose(vmax, vmin):
        vmax = vmin + 1e-12

    levels = np.linspace(vmin, vmax, 61)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for axis, values, title in [
        (axes[0], reference[valid], "Fluent"),
        (axes[1], test[valid], "Panel + ADM"),
    ]:
        axis.tricontourf(
            triangulation,
            values,
            levels=levels,
            cmap=meta["cmap"],
            norm=norm,
        )
        axis.set_aspect("equal")
        axis.set_xlabel(f"{plane_axes[0].upper()} [m]")
        axis.set_ylabel(f"{plane_axes[1].upper()} [m]")
        axis.set_title(f"{title}: {quantity_label(quantity)}")
        axis.grid(alpha=0.2)

    colorbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=meta["cmap"]),
        ax=axes,
        label=quantity_label(quantity),
    )
    colorbar.set_ticks(np.linspace(vmin, vmax, 7))
    fig.suptitle(f"ADM Cut-Plane Comparison ({plane_title})")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if not show_plots:
        plt.close(fig)


def _plot_difference(
    x: np.ndarray,
    y: np.ndarray,
    quantity: str,
    plane_axes: tuple[str, str],
    plane_title: str,
    difference: np.ndarray,
    output_path: Path,
    show_plots: bool,
) -> None:
    """Plot the panel-minus-reference cut-plane difference."""
    valid = ~np.isnan(difference)
    if np.count_nonzero(valid) < 3:
        return

    meta = quantity_meta(quantity)
    triangulation = mtri.Triangulation(x[valid], y[valid])
    limit = float(np.nanmax(np.abs(difference[valid])))
    if limit <= 0.0:
        limit = 1e-12
    levels = np.linspace(-limit, limit, 61)
    norm = colors.Normalize(vmin=-limit, vmax=limit)

    fig, axis = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    axis.tricontourf(
        triangulation,
        difference[valid],
        levels=levels,
        cmap=meta["difference_cmap"],
        norm=norm,
    )
    axis.set_aspect("equal")
    axis.set_xlabel(f"{plane_axes[0].upper()} [m]")
    axis.set_ylabel(f"{plane_axes[1].upper()} [m]")
    axis.set_title(f"Difference: {quantity_label(quantity)} (Panel + ADM - Fluent)\n{plane_title}")
    axis.grid(alpha=0.2)
    colorbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=meta["difference_cmap"]),
        ax=axis,
        label=quantity_label(quantity),
    )
    colorbar.set_ticks(np.linspace(-limit, limit, 7))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if not show_plots:
        plt.close(fig)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="ADM cut-plane comparison against direct CFD export")
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
        "--volume-field",
        type=Path,
        default=None,
        help="Override path to saved panel volume field (.vts)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <case>/out/validation/adm/cut_plane)",
    )
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    if not args.show_plots:
        matplotlib.use("Agg")

    run_cut_plane_comparison(
        case_dir=args.case_dir,
        mesh_level=args.mesh_level,
        quantities=args.quantities,
        cut_data_path=args.cut_data,
        volume_field_path=args.volume_field,
        output_dir=args.output_dir,
        show_plots=args.show_plots,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
