"""
Solver comparison visualization.

Generate side-by-side and overlay plots comparing surface quantities
(Vt, Cp) across multiple solver formulations. Uses the existing
surface_envelope plotting functions for envelope overlays and adds
additional comparison-specific plots (difference, arc-length line).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from visualization.surface_envelope import (
    compute_outward_normals,
    plot_dual_surface_envelope,
    plot_surface_envelope_comparison,
)

if TYPE_CHECKING:
    from solvers.comparison import ComparisonResult


# Default colour palette (colourblind-friendly, matching project style)
DEFAULT_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]


class SolverComparisonVisualizer:
    """Produce comparison plots from a ComparisonResult.

    Usage::

        from solvers.comparison import SolverComparisonRunner
        from visualization.solver_comparison import SolverComparisonVisualizer

        runner = SolverComparisonRunner(case)
        result = runner.run(["constant_source", "linear_source"])

        viz = SolverComparisonVisualizer(result, output_dir=case.output_dir)
        viz.plot_all(show=True)
    """

    def __init__(
        self,
        result: "ComparisonResult",
        output_dir: Optional[Path] = None,
        colors: Optional[List[str]] = None,
    ) -> None:
        self.result = result
        self.output_dir = Path(output_dir) if output_dir else None
        self.colors = colors or DEFAULT_COLORS[: len(result.results)]

    # ── public API ──────────────────────────────────────────────────────

    def plot_all(
        self,
        *,
        show: bool = False,
        save: bool = True,
        dpi: int = 150,
        envelope_scale: float = 0.3,
    ) -> Dict[str, Figure]:
        """Generate all standard comparison plots.

        Args:
            show: Display plots interactively.
            save: Save plots to output_dir.
            dpi: Resolution for saved PNG files.
            envelope_scale: Scale factor for envelope displacement.

        Returns:
            Dict mapping plot name → Figure.
        """
        figures: Dict[str, Figure] = {}

        figures["vt_envelope"] = self.plot_vt_envelope(scale=envelope_scale)
        figures["cp_envelope"] = self.plot_cp_envelope(scale=envelope_scale)
        figures["vt_arc_length"] = self.plot_vt_vs_arc_length()
        figures["cp_arc_length"] = self.plot_cp_vs_arc_length()

        if len(self.result.results) == 2:
            figures["vt_dual"] = self.plot_vt_dual_envelope(scale=envelope_scale)
            figures["vt_difference"] = self.plot_vt_difference()

        if self.result.metrics:
            figures["metrics_table"] = self.plot_metrics_table()

        if save and self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            for name, fig in figures.items():
                path = self.output_dir / f"solver_cmp_{name}.png"
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                print(f"  Saved: {path}")

        if show:
            plt.show()
        else:
            for fig in figures.values():
                plt.close(fig)

        return figures

    # ── envelope plots ──────────────────────────────────────────────────

    def plot_vt_envelope(self, scale: float = 0.3) -> Figure:
        """Overlay Vt surface envelopes from all solvers."""
        r = self.result
        fig, ax = plot_surface_envelope_comparison(
            x_list=r.x_list,
            y_list=r.y_list,
            values_list=r.vt_list,
            labels=r.labels,
            scale=scale,
            quantity_name="Vt",
            colors=self.colors,
            title=f"Tangential Velocity Envelope — {r.case.name} "
                  f"({r.case.num_panels} panels)",
        )
        return fig

    def plot_cp_envelope(self, scale: float = 0.3) -> Figure:
        """Overlay Cp surface envelopes from all solvers (inverted: suction outward)."""
        r = self.result
        fig, ax = plot_surface_envelope_comparison(
            x_list=r.x_list,
            y_list=r.y_list,
            values_list=r.cp_list,
            labels=r.labels,
            scale=scale,
            quantity_name="Cp",
            colors=self.colors,
            invert_values=True,
            title=f"Pressure Coefficient Envelope — {r.case.name} "
                  f"({r.case.num_panels} panels)",
        )
        return fig

    def plot_vt_dual_envelope(self, scale: float = 0.3) -> Figure:
        """Dual envelope for exactly two solvers (same body, two envelopes)."""
        if len(self.result.results) < 2:
            raise ValueError("Dual envelope requires at least 2 solver results")

        r0, r1 = self.result.results[0], self.result.results[1]
        fig, ax = plot_dual_surface_envelope(
            x=r0.surface.x,
            y=r0.surface.y,
            values1=r0.surface.Vt,
            values2=r1.surface.Vt,
            label1=r0.label,
            label2=r1.label,
            scale=scale,
            quantity_name="Vt",
            color1=self.colors[0],
            color2=self.colors[1],
            show_difference=True,
            title=f"Vt Dual Envelope — {self.result.case.name}",
        )
        return fig

    # ── line plots (Vt / Cp vs arc length) ──────────────────────────────

    def plot_vt_vs_arc_length(self) -> Figure:
        """Plot Vt vs arc-length s for all solvers on a standard line chart."""
        return self._plot_quantity_vs_arc_length(
            quantity="Vt",
            ylabel=r"$V_t / V_\infty$",
            title=f"Tangential Velocity vs Arc Length — {self.result.case.name}",
        )

    def plot_cp_vs_arc_length(self) -> Figure:
        """Plot Cp vs arc-length s for all solvers."""
        return self._plot_quantity_vs_arc_length(
            quantity="Cp",
            ylabel=r"$C_p$",
            title=f"Pressure Coefficient vs Arc Length — {self.result.case.name}",
            invert_yaxis=True,
        )

    # ── difference plot ─────────────────────────────────────────────────

    def plot_vt_difference(self) -> Figure:
        """Plot Vt difference between first two solvers vs arc length."""
        if len(self.result.results) < 2:
            raise ValueError("Difference plot requires at least 2 solver results")

        r0, r1 = self.result.results[0], self.result.results[1]
        s = r0.surface.s
        diff = r1.surface.Vt - r0.surface.Vt

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})

        # Top: both Vt curves
        axes[0].plot(s, r0.surface.Vt, color=self.colors[0], label=r0.label, linewidth=1.5)
        axes[0].plot(s, r1.surface.Vt, color=self.colors[1], label=r1.label, linewidth=1.5)
        axes[0].set_ylabel(r"$V_t / V_\infty$")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(f"Vt Comparison — {self.result.case.name}")

        # Bottom: difference
        axes[1].fill_between(s, diff, 0, alpha=0.4, color=self.colors[1])
        axes[1].plot(s, diff, color=self.colors[1], linewidth=1.0)
        axes[1].axhline(0, color="k", linewidth=0.5)
        axes[1].set_xlabel("Arc length s")
        axes[1].set_ylabel(r"$\Delta V_t$")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    # ── metrics table ───────────────────────────────────────────────────

    def plot_metrics_table(self) -> Figure:
        """Render comparison metrics as a matplotlib table figure."""
        metrics = self.result.metrics
        if not metrics:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "No metrics available", ha="center", va="center")
            ax.axis("off")
            return fig

        # Build table data
        col_labels = ["Pair", "Vt L∞", "Vt RMS", "Vt MAE", "Vt rel%",
                       "Cp L∞", "Cp RMS", "Cp MAE", "Cp rel%"]
        rows = []
        for pair, m in metrics.items():
            rows.append([
                pair,
                f"{m['Vt_Linf']:.5f}",
                f"{m['Vt_L2']:.5f}",
                f"{m['Vt_MAE']:.5f}",
                f"{m['Vt_rel_L2_pct']:.2f}%",
                f"{m['Cp_Linf']:.5f}",
                f"{m['Cp_L2']:.5f}",
                f"{m['Cp_MAE']:.5f}",
                f"{m['Cp_rel_L2_pct']:.2f}%",
            ])

        fig, ax = plt.subplots(figsize=(14, 1.5 + 0.5 * len(rows)))
        ax.axis("off")
        table = ax.table(
            cellText=rows,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.4)
        ax.set_title(f"Solver Comparison Metrics — {self.result.case.name}",
                     fontsize=11, pad=12)
        fig.tight_layout()
        return fig

    # ── internal helpers ────────────────────────────────────────────────

    def _plot_quantity_vs_arc_length(
        self,
        quantity: str,
        ylabel: str,
        title: str,
        invert_yaxis: bool = False,
    ) -> Figure:
        """Generic arc-length line plot for a named surface quantity."""
        fig, ax = plt.subplots(figsize=(10, 5))

        for i, r in enumerate(self.result.results):
            s = r.surface.s
            vals = getattr(r.surface, quantity)
            ax.plot(s, vals, color=self.colors[i], label=r.label, linewidth=1.5)

        ax.set_xlabel("Arc length s")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(title)

        if invert_yaxis:
            ax.invert_yaxis()

        fig.tight_layout()
        return fig
