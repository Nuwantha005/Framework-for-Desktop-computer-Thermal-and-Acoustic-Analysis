"""Boundary-layer line and integral-quantity comparison plots."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from visualization.bl_envelope_plots import plot_bl_envelope_comparison
from visualization.bl_plot_common import _LABELS, _color_for


def plot_bl_line(
    path_result,
    quantity: str = "cf",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, Axes]:
    """Plot a BL quantity vs arc length for one side, with all profiles."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    for idx, (name, res) in enumerate(path_result.results.items()):
        vals = getattr(res, quantity)
        color = _color_for(name, idx)
        ax.plot(res.s, vals, color=color, linewidth=1.5, label=name)

        tr = path_result.transitions.get(name)
        if tr is not None and tr.transition_s is not None:
            s_tr = tr.transition_s
            vi = np.searchsorted(res.s, s_tr)
            vi = min(vi, len(vals) - 1)
            ax.axvline(s_tr, color=color, ls="--", alpha=0.4)
            ax.plot(s_tr, vals[vi], "o", color=color, ms=6)

    ax.set_xlabel("Arc length $s$ [m]")
    ax.set_ylabel(_LABELS.get(quantity, quantity))
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_bl_lines_multi(
    path_result,
    quantities: Optional[List[str]] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, List[Axes]]:
    """Multi-panel line plots of several BL quantities for one side."""
    if quantities is None:
        quantities = ["cf", "delta_star", "theta", "H"]
    n_quantities = len(quantities)

    fig, axes = plt.subplots(n_quantities, 1, figsize=(8, 3.2 * n_quantities), sharex=True)
    if n_quantities == 1:
        axes = [axes]

    for ax, quantity in zip(axes, quantities):
        plot_bl_line(path_result, quantity=quantity, ax=ax)

    axes[-1].set_xlabel("Arc length $s$ [m]")
    if title:
        fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, list(axes)


def plot_bl_two_sides(
    case_result,
    quantities: Optional[List[str]] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, NDArray]:
    """Side-by-side line plots: upper (left) and lower (right)."""
    if quantities is None:
        quantities = ["cf", "delta_star", "theta", "H"]
    n_quantities = len(quantities)

    fig, axes = plt.subplots(n_quantities, 2, figsize=(14, 3.2 * n_quantities), sharex="col")
    if n_quantities == 1:
        axes = axes.reshape(1, 2)

    for i, quantity in enumerate(quantities):
        plot_bl_line(case_result.upper, quantity=quantity, ax=axes[i, 0])
        plot_bl_line(case_result.lower, quantity=quantity, ax=axes[i, 1])
        axes[i, 1].set_ylabel("")

    axes[0, 0].set_title("Upper side", fontsize=12)
    axes[0, 1].set_title("Lower side", fontsize=12)
    axes[-1, 0].set_xlabel("Arc length $s$ [m]")
    axes[-1, 1].set_xlabel("Arc length $s$ [m]")

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, axes


def plot_bl_comparison(
    case_result,
    quantities: Optional[List[str]] = None,
    envelope_quantity: str = "cf",
    envelope_scale: float = 0.15,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> Figure:
    """Full comparison figure: two-sided lines + envelope + Ue."""
    if quantities is None:
        quantities = ["cf", "delta_star", "theta", "H"]

    n_rows = max(len(quantities), 2)
    fig = plt.figure(figsize=(18, 3.5 * n_rows))
    gs = fig.add_gridspec(n_rows, 3, width_ratios=[1, 1, 1.2], hspace=0.35, wspace=0.30)

    for i, quantity in enumerate(quantities):
        ax_upper = fig.add_subplot(gs[i, 0])
        plot_bl_line(case_result.upper, quantity=quantity, ax=ax_upper)
        if i == 0:
            ax_upper.set_title("Upper side", fontsize=11)
        if i < len(quantities) - 1:
            ax_upper.set_xlabel("")

        ax_lower = fig.add_subplot(gs[i, 1])
        plot_bl_line(case_result.lower, quantity=quantity, ax=ax_lower)
        ax_lower.set_ylabel("")
        if i == 0:
            ax_lower.set_title("Lower side", fontsize=11)
        if i < len(quantities) - 1:
            ax_lower.set_xlabel("")

    ax_env = fig.add_subplot(gs[: n_rows - 1, 2])
    plot_bl_envelope_comparison(
        case_result,
        quantity=envelope_quantity,
        scale=envelope_scale,
        ax=ax_env,
        title=f"{envelope_quantity} envelope",
    )

    ax_ue = fig.add_subplot(gs[n_rows - 1, 2])
    ax_ue.plot(case_result.upper.s, case_result.upper.Ue, "b-", lw=1.5, label="Upper $U_e$")
    ax_ue.plot(case_result.lower.s, case_result.lower.Ue, "r-", lw=1.5, label="Lower $U_e$")
    ax_ue.set_xlabel("Arc length $s$ [m]")
    ax_ue.set_ylabel("$U_e$ [m/s]")
    ax_ue.legend(fontsize=9)
    ax_ue.grid(True, alpha=0.3)
    ax_ue.set_title("Edge velocity (panel method)")

    if title:
        fig.suptitle(title, fontsize=15, y=1.02)

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
