"""Tests for actuator disk model utilities."""

from pathlib import Path

import numpy as np

from core.io import CaseLoader
from solvers.actuator import FanCurve, generate_actuator_disk_mesh
from solvers.actuator.doublet_influence import (
    compute_point_doublet_velocity,
    integrate_flow_rate,
    pressure_jump_to_doublet_strength,
)
from solvers.actuator.coupling import ActuatorDiskCoupledSolver3D
from solvers.panel3d import SourcePanelSolver3D


def test_fan_curve_loads_and_interpolates_circular_vent_curve():
    curve = FanCurve.from_csv(
        Path("cases/cicular_vent/data/fan_curve.csv"),
        interpolation="linear",
    )

    assert curve.flow_rate.size >= 2
    assert curve.pressure_at(float(curve.flow_rate[0])) == float(curve.pressure[0])

    q_mid = 0.5 * (float(curve.flow_rate[0]) + float(curve.flow_rate[1]))
    dp_mid = curve.pressure_at(q_mid)
    assert min(curve.pressure[0], curve.pressure[1]) <= dp_mid <= max(curve.pressure[0], curve.pressure[1])


def test_actuator_disk_mesh_area_and_orientation():
    mesh = generate_actuator_disk_mesh(
        center=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        radius=0.06,
        n_r=8,
        n_theta=64,
    )

    assert mesh.num_panels == 8 * 64
    expected_normals = np.tile(np.array([0.0, 0.0, 1.0]), (mesh.num_panels, 1))
    np.testing.assert_allclose(mesh.normals, expected_normals, atol=1e-12)
    assert np.isclose(np.sum(mesh.areas), np.pi * 0.06**2, rtol=2e-3)


def test_doublet_strength_mapping_and_flow_integration():
    mu = pressure_jump_to_doublet_strength(
        pressure_rise=12.25,
        density=1.225,
        reference_velocity=2.0,
        characteristic_length=0.1,
    )
    assert np.isclose(mu, 0.5)

    mesh = generate_actuator_disk_mesh(
        center=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        radius=0.5,
        n_r=2,
        n_theta=16,
    )
    normal_velocity = np.full(mesh.num_panels, 3.0)
    assert np.isclose(integrate_flow_rate(normal_velocity, mesh), 3.0 * np.sum(mesh.areas))

    velocity = compute_point_doublet_velocity(
        points=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        disk_mesh=mesh,
        doublet_strength=np.full(mesh.num_panels, mu),
    )
    assert velocity.shape == (1, 3)
    assert np.all(np.isfinite(velocity))


def test_cases_without_actuator_disks_use_plain_3d_solver():
    case = CaseLoader.load_case("cases/sphere_flow", mesh_level_index=0)
    solver = case.create_solver()

    assert isinstance(solver, SourcePanelSolver3D)


def test_case_with_actuator_disks_uses_coupled_solver():
    case = CaseLoader.load_case("cases/cicular_vent", mesh_level_index=0)
    solver = case.create_solver()

    assert isinstance(solver, ActuatorDiskCoupledSolver3D)


def test_pressure_reconstruction_uses_disk_pressure_jump():
    case = CaseLoader.load_case("cases/cicular_vent", mesh_level_index=0)
    solver = case.create_solver()

    assert isinstance(solver, ActuatorDiskCoupledSolver3D)
    solver._disks = solver._build_disks()

    disk = solver._disks[0]
    disk.pressure_rise = 20.0

    points = np.array(
        [
            [0.0, 0.0, -0.1],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1],
            [0.2, 0.0, 0.1],
        ],
        dtype=np.float64,
    )
    velocity = np.tile(np.array([0.0, 0.0, 2.0], dtype=np.float64), (4, 1))

    pressure = solver._pressure_from_velocity(points, velocity)
    dynamic = 0.5 * case.density * 2.0**2

    np.testing.assert_allclose(
        pressure,
        np.array(
            [
                case.reference_pressure - dynamic,
                case.reference_pressure - dynamic + 10.0,
                case.reference_pressure - dynamic + 20.0,
                case.reference_pressure - dynamic,
            ],
            dtype=np.float64,
        ),
        atol=1e-10,
    )


if __name__ == "__main__":
    print(
        "This file contains pytest tests. Run them with:\n"
        "  python -m pytest src/test/test_actuator_disk.py -q\n\n"
        "To run the circular vent ADM case and see solver output, use:\n"
        "  python demos/demo_actuator_disk.py --case cases/cicular_vent"
    )
