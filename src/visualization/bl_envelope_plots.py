"""Boundary-layer envelope plots on body geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from visualization.bl_plot_common import _color_for
from visualization.surface_envelope import (
    plot_surface_envelope,
    plot_surface_envelope_comparison,
)


def plot_bl_envelope(
    case_result,
    quantity: str = "cf",
    profile_name: Optional[str] = None,
    scale: float = 0.15,
    colormap: Optional[str] = "magma",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, Axes]:
    """Envelope plot of a BL quantity on the full body for one profile."""
    if profile_name is None:
        profile_name = case_result.profile_names[0]

    vals = case_result.full_body_quantity(quantity, profile_name)
    vals_plot = np.where(np.isnan(vals), 0.0, vals)

    fig, ax = plot_surface_envelope(
        case_result.surface_x,
        case_result.surface_y,
        vals_plot,
        scale=scale,
        quantity_name=f"{quantity} ({profile_name})",
        colormap=colormap,
        ax=ax,
        title=title or f"{quantity} envelope - {profile_name}",
    )

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_bl_envelope_comparison(
    case_result,
    quantity: str = "cf",
    scale: float = 0.15,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, Axes]:
    """Overlay envelope plots of multiple profiles on the full body."""
    x_list, y_list, values_list, labels = [], [], [], []

    for name in case_result.profile_names:
        vals = case_result.full_body_quantity(quantity, name)
        vals_plot = np.where(np.isnan(vals), 0.0, vals)
        x_list.append(case_result.surface_x)
        y_list.append(case_result.surface_y)
        values_list.append(vals_plot)
        labels.append(name)

    colors = [_color_for(name, i) for i, name in enumerate(labels)]

    fig, ax = plot_surface_envelope_comparison(
        x_list,
        y_list,
        values_list,
        labels,
        scale=scale,
        quantity_name=quantity,
        colors=colors,
        ax=ax,
        title=title or f"{quantity} - profile comparison",
    )

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax
