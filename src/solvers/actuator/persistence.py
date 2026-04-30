"""Persistence helpers for ADM solver runs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .coupling import ActuatorDiskCoupledSolver3D


def save_adm_solver_run(path: str | Path, solver: "ActuatorDiskCoupledSolver3D") -> None:
    """Save a compact ADM solution bundle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    disk_names = np.array([disk.config.name for disk in solver._disks])
    disk_offsets = []
    disk_nodes = []
    disk_panels = []
    disk_mu = []
    disk_vn = []
    node_offset = 0
    panel_offset = 0
    for disk in solver._disks:
        disk_offsets.append((node_offset, panel_offset, disk.mesh.num_panels))
        disk_nodes.append(disk.mesh.nodes)
        disk_panels.append(disk.mesh.panels + node_offset)
        disk_mu.append(disk.doublet_strength)
        disk_vn.append(disk.normal_velocity)
        node_offset += disk.mesh.nodes.shape[0]
        panel_offset += disk.mesh.num_panels

    history = np.array([
        (
            item.iteration,
            item.disk_name,
            item.flow_rate,
            item.pressure_rise,
            item.pressure_rise_curve,
            item.pressure_residual,
        )
        for item in solver.convergence_history
    ], dtype=object)

    np.savez_compressed(
        path,
        freestream=solver._v_inf,
        density=np.array([solver._density], dtype=np.float64),
        body_sigma=getattr(solver.body_solver, "sigma", np.array([], dtype=np.float64)),
        body_velocity=solver.surface_velocity,
        disk_names=disk_names,
        disk_offsets=np.asarray(disk_offsets, dtype=np.int32),
        disk_nodes=np.vstack(disk_nodes) if disk_nodes else np.zeros((0, 3)),
        disk_panels=np.vstack(disk_panels) if disk_panels else np.zeros((0, 4), dtype=np.int32),
        disk_doublet_strength=np.concatenate(disk_mu) if disk_mu else np.zeros(0),
        disk_normal_velocity=np.concatenate(disk_vn) if disk_vn else np.zeros(0),
        result_flow_rate=np.array([result.flow_rate for result in solver._results]),
        result_pressure_rise=np.array([result.pressure_rise for result in solver._results]),
        history=history,
    )
