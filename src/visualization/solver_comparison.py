"""
Solver comparison visualization.

Generate side-by-side and overlay plots comparing surface quantities
(Vt, Cp) across multiple panel method solvers and optional OpenFOAM
reference data. Uses the existing surface_envelope plotting functions
for envelope overlays and adds comparison-specific plots (difference,
arc-length line, metrics table, ranking).
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
    from solvers.comparison import ComparisonResult, SolverResult


# Default colour palette (colourblind-friendly, matching project style)
DEFAULT_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
REFERENCE_COLOR = "#333333"  # dark grey for OpenFOAM reference
REFERENCE_STYLE: Dict = dict(color=REFERENCE_COLOR, linewidth=2.0, linestyle="--", alpha=0.85)

# Default subfolder name for solver comparisons
COMPARISON_SUBFOLDER = "solver_comparison"


class SolverComparisonVisualizer:
    """Produce comparison plots from a ComparisonResult.

    Output is saved to ``<output_dir>/solver_comparison/`` by default.
    When an OpenFOAM reference is present in the result, it is drawn as
    a dashed dark-grey line in every plot so panel solvers can be compared
    visually against CFD.

    Usage::

        from solvers.comparison import SolverComparisonRunner
        from visualization.solver_comparison import SolverComparisonVisualizer

        runner = SolverComparisonRunner(case)
        result = runner.run(["constant", "linear"],
                            of_case_dir=Path("cases/rounded_square/of_case/cases/level_4"))

        viz = SolverComparisonVisualizer(result, output_dir=case.output_dir)
        viz.plot_all(show=True)
    """

    def __init__(
        self,
        result: "ComparisonResult",
        output_dir: Optional[Path] = None,
        colors: Optional[List[str]] = None,
        *,
        subfolder: str = COMPARISON_SUBFOLDER,
    ) -> None:
        self.result = result
        # Resolve output directory — always use a subfolder
        if output_dir is not None:
            base = Path(output_dir)
            self.output_dir = base / subfolder if subfolder else base
        else:
            self.output_dir = None

        # Assign colours — skip reference (it uses REFERENCE_STYLE)
        solver_count = len(result.solver_results) if hasattr(result, "solver_results") else len(result.results)
        self.colors = colors or DEFAULT_COLORS[:solver_count]

    # ── helpers ─────────────────────────────────────────────────────────

    def _has_reference(self) -> bool:
        return self.result.reference is not None

    def _solver_results(self) -> List["SolverResult"]:
        return self.result.solver_results

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
            Dict mapping plot name -> Figure.
        """
        figures: Dict[str, Figure] = {}

        figures["vt_envelope"] = self.plot_vt_envelope(scale=envelope_scale)
        figures["cp_envelope"] = self.plot_cp_envelope(scale=envelope_scale)
        figures["vt_arc_length"] = self.plot_vt_vs_arc_length()
        figures["cp_arc_length"] = self.plot_cp_vs_arc_length()

        solver_results = self._solver_results()
        if len(solver_results) == 2:
            figures["vt_dual"] = self.plot_vt_dual_envelope(scale=envelope_scale)
            figures["vt_difference"] = self.plot_vt_difference()

        if self.result.metrics:
            figures["metrics_table"] = self.plot_metrics_table()

        if self.result.ranking:
            figures["ranking"] = self.plot_ranking()

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
        """Overlay Vt surface envelopes from all solvers (+ OF reference)."""
        r = self.result
        fig, ax = plot_surface_envelope_comparison(
            x_list=r.x_list,
            y_list=r.y_list,
            values_list=r.vt_list,
            labels=r.labels,
            scale=scale,
            quantity_name="Vt",
            colors=self._all_colors(),
            title=self._title("Tangential Velocity Envelope"),
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
            colors=self._all_colors(),
            invert_values=True,
            title=self._title("Pressure Coefficient Envelope"),
        )
        return fig

    def plot_vt_dual_envelope(self, scale: float = 0.3) -> Figure:
        """Dual envelope for exactly two panel-method solvers."""
        solvers = self._solver_results()
        if len(solvers) < 2:
            raise ValueError("Dual envelope requires at least 2 solver results")

        r0, r1 = solvers[0], solvers[1]
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
        """Plot Vt vs arc-length s for all solvers (+ OF reference)."""
        return self._plot_quantity_vs_arc_length(
            quantity="Vt",
            ylabel=r"Normalized Velocity ($V_t / V_\infty$)",
            title=self._title("Normalized Tangential Velocity vs Arc Length"),
        )

    def plot_cp_vs_arc_length(self) -> Figure:
        """Plot Cp vs arc-length s for all solvers (+ OF reference)."""
        return self._plot_quantity_vs_arc_length(
            quantity="Cp",
            ylabel=r"$C_p$",
            title=self._title("Pressure Coefficient vs Arc Length"),
            invert_yaxis=True,
        )

    # ── difference plot ─────────────────────────────────────────────────

    def plot_vt_difference(self) -> Figure:
        """Plot Vt difference between first two panel-method solvers."""
        solvers = self._solver_results()
        if len(solvers) < 2:
            raise ValueError("Difference plot requires at least 2 solver results")

        r0, r1 = solvers[0], solvers[1]
        s = r0.surface.s
        diff = r1.surface.Vt - r0.surface.Vt

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})

        # If reference, draw it on top panel
        ref = self.result.reference
        if ref is not None:
            axes[0].plot(ref.surface.s, ref.surface.Vt, label=ref.label, **REFERENCE_STYLE)

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

        # Highlight best row (lowest Vt rel%) in green
        if self.result.ranking:
            best_label = self.result.ranking[0][0]
            for row_idx, row_data in enumerate(rows):
                if row_data[0].startswith(best_label):
                    for col_idx in range(len(col_labels)):
                        table[row_idx + 1, col_idx].set_facecolor("#d4edda")

        ax.set_title(self._title("Comparison Metrics"),
                     fontsize=11, pad=12)
        fig.tight_layout()
        return fig

    # ── ranking chart ───────────────────────────────────────────────────

    def plot_ranking(self) -> Figure:
        """Horizontal bar chart ranking solvers by Vt relative-L2 vs reference."""
        ranking = self.result.ranking
        if not ranking:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "No ranking available", ha="center", va="center")
            ax.axis("off")
            return fig

        labels = [r[0] for r in ranking]
        values = [r[1] for r in ranking]

        fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(labels) + 1.5)))
        bar_colors = ["#2ca02c" if i == 0 else "#1f77b4" for i in range(len(labels))]
        bars = ax.barh(labels[::-1], values[::-1], color=bar_colors[::-1], edgecolor="white")

        # Annotate values
        for bar, val in zip(bars, values[::-1]):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}%", va="center", fontsize=9)

        ax.set_xlabel("Relative Vt RMS Error (%)")
        ax.set_title(self._title("Solver Ranking vs Reference"))
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return fig

    # ── internal helpers ────────────────────────────────────────────────

    def _title(self, base: str) -> str:
        """Build a plot title with case name and panel count."""
        return f"{base} — {self.result.case.name} ({self.result.case.num_panels} panels)"

    def _all_colors(self) -> List[str]:
        """Return colour list matching all results (reference gets REFERENCE_COLOR)."""
        colors = []
        solver_idx = 0
        for r in self.result.results:
            if r.is_reference:
                colors.append(REFERENCE_COLOR)
            else:
                colors.append(self.colors[solver_idx % len(self.colors)])
                solver_idx += 1
        return colors

    def _plot_quantity_vs_arc_length(
        self,
        quantity: str,
        ylabel: str,
        title: str,
        invert_yaxis: bool = False,
    ) -> Figure:
        """Generic arc-length line plot for a named surface quantity.

        The OpenFOAM reference (if present) is drawn as a dashed grey line.
        Panel-method solvers use solid coloured lines.
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        # Draw reference first (behind solver curves)
        ref = self.result.reference
        if ref is not None:
            s = ref.surface.s
            vals = getattr(ref.surface, quantity)
            ax.plot(s, vals, label=ref.label, **REFERENCE_STYLE)

        # Draw panel-method solver curves
        for i, r in enumerate(self._solver_results()):
            s = r.surface.s
            vals = getattr(r.surface, quantity)
            ax.plot(s, vals, color=self.colors[i % len(self.colors)],
                    label=r.label, linewidth=1.5)

        ax.set_xlabel("Arc length (m)")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(title)

        if invert_yaxis:
            ax.invert_yaxis()

        fig.tight_layout()
        return fig
