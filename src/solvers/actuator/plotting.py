"""Plotting helpers for ADM convergence."""

from __future__ import annotations

from pathlib import Path
import os

from .models import ADMIterationRecord, ActuatorDiskRuntime


def plot_adm_convergence(history: list[ADMIterationRecord], path: str | Path) -> None:
    """Save a pressure-residual convergence plot."""
    if not history:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    names = sorted({item.disk_name for item in history})
    for name in names:
        records = [item for item in history if item.disk_name == name]
        ax.semilogy(
            [item.iteration for item in records],
            [abs(item.pressure_residual) for item in records],
            marker="o",
            label=name,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$|\Delta p_{curve} - \Delta p|$ [Pa]")
    ax.set_title("Actuator Disk P-Q Coupling Convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_fan_curve_progression(
    disks: list[ActuatorDiskRuntime],
    history: list[ADMIterationRecord],
    path: str | Path,
) -> None:
    """Save fan P-Q curves with ADM iteration operating points."""
    if not disks or not history:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    scatter = None
    for disk in disks:
        curve = disk.curve
        ax.plot(
            curve.flow_rate,
            curve.pressure,
            linewidth=2.0,
            label=f"{disk.config.name} P-Q curve",
        )

        records = [item for item in history if item.disk_name == disk.config.name]
        if not records:
            continue
        q_values = [item.flow_rate for item in records]
        dp_values = [item.pressure_rise for item in records]
        iterations = [item.iteration for item in records]
        scatter = ax.scatter(
            q_values,
            dp_values,
            c=iterations,
            cmap="viridis",
            s=38,
            edgecolors="black",
            linewidths=0.35,
            label=f"{disk.config.name} ADM iterates",
            zorder=3,
        )
        ax.plot(q_values, dp_values, color="0.35", linewidth=1.0, alpha=0.75)
        ax.scatter(
            [q_values[-1]],
            [dp_values[-1]],
            marker="x",
            color="red",
            s=90,
            linewidths=2.0,
            label=f"{disk.config.name} final",
            zorder=4,
        )

    ax.set_xlabel(r"Flow rate $Q$ [m$^3$/s]")
    ax.set_ylabel(r"Static pressure rise $\Delta p$ [Pa]")
    ax.set_title("Actuator Disk Operating Point Progression")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if scatter is not None:
        fig.colorbar(scatter, ax=ax, label="Iteration")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
