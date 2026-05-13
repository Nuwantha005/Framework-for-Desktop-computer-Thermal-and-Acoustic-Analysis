"""Coupled 3D panel solver with actuator disks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from core.config.schemas import ActuatorDiskConfig, SolverConfig, BoundaryRegionConfig
from core.geometry.mesh3d import Mesh3D
from solvers.factory import SolverFactory
from solvers.panel3d.base import PanelSolver3D

from .disk_mesh import generate_actuator_disk_mesh
from .doublet_influence import (
    compute_disk_normal_velocity,
    compute_point_doublet_velocity,
    integrate_flow_rate,
    pressure_jump_to_doublet_strength,
)
from .fan_curve import FanCurve
from .models import ADMIterationRecord, ActuatorDiskResult, ActuatorDiskRuntime
from solvers.panel3d.influences.source3d import compute_all_velocities_influence
from .persistence import save_adm_solver_run
from .plotting import plot_adm_convergence, plot_fan_curve_progression

if TYPE_CHECKING:
    from core.io.case import Case



class BoundaryRegionRuntime:
    config: BoundaryRegionConfig
    mesh: Mesh3D
    source_strength: NDArray[np.float64]
    area: float
    
    def __init__(self, config, mesh):
        self.config = config
        self.mesh = mesh
        self.source_strength = np.zeros(mesh.num_panels, dtype=np.float64)
        self.area = float(np.sum(mesh.areas))

class ActuatorDiskCoupledSolver3D:
    """Couple configured 3D panel solvers with actuator disk pressure jumps."""

    def __init__(
        self,
        mesh: Mesh3D,
        v_inf: NDArray[np.float64],
        density: float,
        reference_pressure: float,
        disk_configs: list[ActuatorDiskConfig],
        case_dir: Path,
        solver_config: SolverConfig,
        inlets: list[BoundaryRegionConfig] = None,
        outlets: list[BoundaryRegionConfig] = None,
    ) -> None:
        """Initialize the coupled ADM solver."""
        if mesh.dimension != 3:
            raise ValueError("ActuatorDiskCoupledSolver3D requires a 3D mesh")
        self._mesh = mesh
        self._v_inf = np.asarray(v_inf, dtype=np.float64)
        self._density = float(density)
        self._reference_pressure = float(reference_pressure)
        self._disk_configs = disk_configs
        self._case_config_inlets = inlets or []
        self._case_config_outlets = outlets or []
        self._case_dir = Path(case_dir)
        self._solver_config = solver_config
        self._body_solver: PanelSolver3D | None = None
        self._disks: list[ActuatorDiskRuntime] = []
        self._inlets: list[BoundaryRegionRuntime] = []
        self._outlets: list[BoundaryRegionRuntime] = []
        self.system_q = 0.0
        self._history: list[ADMIterationRecord] = []
        self._results: list[ActuatorDiskResult] = []
        self._warnings: dict[str, str] = {}
        self._solved = False

    @classmethod
    def from_case(cls, case: "Case") -> "ActuatorDiskCoupledSolver3D":
        """Create a coupled ADM solver from a case."""
        return cls(
            mesh=case.mesh,
            v_inf=case.freestream,
            density=case.density,
            reference_pressure=case.reference_pressure,
            disk_configs=case.config.actuator_disks,
            case_dir=case.case_dir,
            solver_config=case.config.solver,
            inlets=case.config.inlets,
            outlets=case.config.outlets,
        )

    @property
    def mesh(self) -> Mesh3D:
        """Body mesh."""
        return self._mesh

    @property
    def body_solver(self) -> PanelSolver3D:
        """Configured body panel solver."""
        if self._body_solver is None:
            raise RuntimeError("Solver not executed. Call solve() first.")
        return self._body_solver

    @property
    def actuator_results(self) -> list[ActuatorDiskResult]:
        """Final actuator disk operating points."""
        if not self._solved:
            raise RuntimeError("Solver not executed. Call solve() first.")
        return self._results

    @property
    def convergence_history(self) -> list[ADMIterationRecord]:
        """ADM iteration records."""
        return self._history

    @property
    def surface_velocity(self) -> NDArray[np.float64]:
        """Velocity at body panel centers."""
        return self.body_solver.surface_velocity

    @property
    def is_solved(self) -> bool:
        """Whether the coupled solve has completed."""
        return self._solved

    @property
    def Cp(self) -> NDArray[np.float64]:
        """Body pressure coefficient from the wrapped solver."""
        return self.body_solver.Cp

    @property
    def sigma(self) -> NDArray[np.float64]:
        """Body source strengths when available."""
        return self.body_solver.sigma

    @property
    def reference_pressure(self) -> float:
        """Reference static pressure used for Bernoulli reconstruction."""
        return self._reference_pressure

    def solve(self) -> None:
        """Run the P-Q actuator disk coupling loop."""
        if not self._disk_configs:
            self._body_solver = self._create_body_solver()
            self._body_solver.solve()
            self._solved = True
            return

        self._disks = self._build_disks()
        self._inlets = [self._build_boundary(cfg) for cfg in self._case_config_inlets]
        self._outlets = [self._build_boundary(cfg) for cfg in self._case_config_outlets]
        converged = False
        max_iterations = max(disk.config.max_iterations for disk in self._disks)

        for iteration in range(1, max_iterations + 1):
            # Total intake flow rate is the sum of system_q for all intake fans.
            # We assume fans with normal pointing in -y or -z or -x into the casing are intakes.
            # Or simpler: just sum the flows of all fans that pull air from outside.
            # Let's just sum all disk.system_q that have "intake" in their name, or if not found, use the first disk's q.
            intake_q = sum(d.system_q for d in self._disks if "intake" in d.config.name.lower())
            if intake_q == 0 and self._disks:
                intake_q = self._disks[0].system_q
                
            total_inlet_area = sum(inlet.area for inlet in self._inlets)
            for inlet in self._inlets:
                if inlet.area > 0 and total_inlet_area > 0:
                    # distribute total intake_q proportionally by area
                    inlet.source_strength = np.full(inlet.mesh.num_panels, 2.0 * intake_q / total_inlet_area)
            
            total_outlet_area = sum(outlet.area for outlet in self._outlets)
            for outlet in self._outlets:
                if outlet.area > 0 and total_outlet_area > 0:
                    outlet.source_strength = np.full(outlet.mesh.num_panels, -2.0 * intake_q / total_outlet_area)
                    
            for disk in self._disks:
                dp_curve = disk.curve.pressure_at(disk.system_q)
                disk.pressure_rise = dp_curve
                self._update_disk_doublet_strength(disk)

            disturbance = self._compute_body_normal_disturbance()
            self._body_solver = self._create_body_solver()
            self._body_solver.solve(normal_velocity_disturbance=disturbance)

            residuals = []
            bounds_reached = False
            
            for disk in self._disks:
                disk_velocity = self._velocity_at_disk(disk)
                disk.normal_velocity = compute_disk_normal_velocity(disk_velocity, disk.mesh)
                measured_q = integrate_flow_rate(disk.normal_velocity, disk.mesh)
                disk.flow_rate = measured_q
                
                if not disk.curve.contains_flow_rate(disk.system_q):
                    bounds_reached = True
                    warning = (
                        f"Fan '{disk.config.name}' flow rate {disk.system_q:.6e} m^3/s "
                        f"left fan-curve range [{disk.curve.q_min:.6e}, "
                        f"{disk.curve.q_max:.6e}] m^3/s; stopping ADM iteration."
                    )
                    self._warnings[disk.config.name] = warning
                    print(f"[ADM WARNING] {warning}")
                    
                dp_curve = disk.curve.pressure_at(disk.system_q)
                residual = measured_q - disk.system_q
                residuals.append(abs(residual))
                self._history.append(
                    ADMIterationRecord(
                        iteration=iteration,
                        disk_name=disk.config.name,
                        flow_rate=measured_q,
                        pressure_rise=dp_curve,
                        pressure_rise_curve=dp_curve,
                        pressure_residual=residual,
                    )
                )
                print(
                    f"[ADM] iter={iteration:03d} fan={disk.config.name} "
                    f"Q_sys={disk.system_q:.6e} m^3/s Q_meas={measured_q:.6e} m^3/s "
                    f"dp={dp_curve:.6e} Pa residual={residual:.6e} m^3/s"
                )

                if not bounds_reached and iteration < disk.config.max_iterations:
                    disk.system_q += disk.config.relaxation * residual

            if bounds_reached:
                break

            active_residuals = [
                residual
                for i, residual in enumerate(residuals)
                if iteration <= self._disks[i].config.max_iterations
            ]
            
            if active_residuals and all(residual <= 1e-5 for residual in active_residuals):
                converged = True
                break

        self._results = [
            ActuatorDiskResult(
                name=disk.config.name,
                flow_rate=disk.flow_rate,
                pressure_rise=disk.pressure_rise,
                doublet_strength=disk.doublet_strength.copy(),
                normal_velocity=disk.normal_velocity.copy(),
                converged=converged,
                iterations=self._history[-1].iteration if self._history else 0,
                warning=self._warnings.get(disk.config.name),
            )
            for disk in self._disks
        ]
        self._write_outputs()
        self._solved = True

    def velocity_at(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute velocity from body and actuator disks at arbitrary points."""
        if not self._solved:
            raise RuntimeError("Solver not executed. Call solve() first.")
        velocity = self.body_solver.velocity_at(points)
        for disk in self._disks:
            velocity += compute_point_doublet_velocity(points, disk.mesh, disk.doublet_strength)
        for inlet in self._inlets:
            velocity += compute_all_velocities_influence(points, inlet.mesh.nodes, inlet.mesh.panels, inlet.source_strength)
        for outlet in self._outlets:
            velocity += compute_all_velocities_influence(points, outlet.mesh.nodes, outlet.mesh.panels, outlet.source_strength)
        return velocity

    def pressure_at(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute static pressure at arbitrary points.

        The reconstruction uses Bernoulli within each region and adds the
        prescribed actuator-disk pressure jumps on the downstream side of each
        disk. Points exactly on a disk receive half the jump so line/cut-plane
        samples remain well-behaved.
        """
        if not self._solved:
            raise RuntimeError("Solver not executed. Call solve() first.")
        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        velocity = self.velocity_at(points)
        velocity = self._regularize_disk_plane_velocity(points, velocity)
        return self._pressure_from_velocity(points, velocity)

    def pressure_gauge_at(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute gauge pressure at arbitrary points."""
        return self.pressure_at(points) - self._reference_pressure

    def _create_body_solver(self) -> PanelSolver3D:
        return SolverFactory.create(
            config=self._solver_config,
            mesh=self._mesh,
            v_inf=self._v_inf,
            aoa=0.0,
        )

    def _build_boundary(self, config: BoundaryRegionConfig) -> BoundaryRegionRuntime:
        normal = np.asarray(config.normal, dtype=np.float64)
        normal = normal / np.linalg.norm(normal)
        if config.shape == "circle":
            from .disk_mesh import generate_actuator_disk_mesh
            mesh = generate_actuator_disk_mesh(
                center=config.center,
                normal=normal,
                radius=config.radius,
                n_r=config.n_r,
                n_theta=config.n_theta,
            )
        elif config.shape == "rectangle":
            from .disk_mesh import generate_rectangular_boundary_mesh
            mesh = generate_rectangular_boundary_mesh(
                center=config.center,
                normal=normal,
                width=config.width,
                height=config.height,
                n_w=config.n_r,  # Reuse n_r for width subdivision
                n_h=config.n_theta, # Reuse n_theta for height subdivision
            )
        else:
            raise NotImplementedError(f"Boundary shape '{config.shape}' not supported")
        return BoundaryRegionRuntime(config, mesh)

    def _build_disks(self) -> list[ActuatorDiskRuntime]:
        disks = []
        for config in self._disk_configs:
            curve_path = self._case_dir / config.curve_file
            curve = FanCurve.from_csv(curve_path, interpolation=config.interpolation)
            normal = np.asarray(config.normal, dtype=np.float64)
            normal = normal / np.linalg.norm(normal)
            mesh = generate_actuator_disk_mesh(
                center=config.center,
                normal=normal,
                radius=config.radius,
                n_r=config.n_r,
                n_theta=config.n_theta,
            )
            pressure_rise = config.dp_initial
            if pressure_rise is None:
                pressure_rise = curve.midpoint_pressure
            disk_area = float(np.sum(mesh.areas))
            q_mid = 0.5 * (curve.q_min + curve.q_max)
            freestream_normal_speed = abs(float(np.dot(self._v_inf, normal)))
            curve_velocity_scale = abs(q_mid / disk_area) if disk_area > 0 else 0.0
            reference_velocity = max(freestream_normal_speed, curve_velocity_scale, 1e-6)
            sample_offset = float(config.radius)
            disk = ActuatorDiskRuntime(
                config=config,
                mesh=mesh,
                curve=curve,
                normal=normal,
                pressure_rise=float(pressure_rise),
                doublet_strength=np.zeros(mesh.num_panels, dtype=np.float64),
                reference_velocity=reference_velocity,
                sample_offset=sample_offset,
            )
            self._update_disk_doublet_strength(disk)
            disks.append(disk)
        return disks

    def _update_disk_doublet_strength(self, disk: ActuatorDiskRuntime) -> None:
        mu = pressure_jump_to_doublet_strength(
            pressure_rise=disk.pressure_rise,
            density=self._density,
            reference_velocity=disk.reference_velocity,
            characteristic_length=disk.config.radius,
        )
        disk.doublet_strength = np.full(disk.mesh.num_panels, mu, dtype=np.float64)

    def _compute_body_normal_disturbance(self) -> NDArray[np.float64]:
        disturbance_velocity = np.zeros((self._mesh.num_panels, 3), dtype=np.float64)
        for disk in self._disks:
            disturbance_velocity += compute_point_doublet_velocity(
                self._mesh.centers,
                disk.mesh,
                disk.doublet_strength,
            )
        for inlet in self._inlets:
            disturbance_velocity += compute_all_velocities_influence(
                self._mesh.centers, inlet.mesh.nodes, inlet.mesh.panels, inlet.source_strength
            )
        for outlet in self._outlets:
            disturbance_velocity += compute_all_velocities_influence(
                self._mesh.centers, outlet.mesh.nodes, outlet.mesh.panels, outlet.source_strength
            )
        return np.einsum("ij,ij->i", disturbance_velocity, self._mesh.normals)

    def _pressure_from_velocity(
        self,
        points: NDArray[np.float64],
        velocity: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Reconstruct static pressure from velocity and disk pressure jumps."""
        velocity = np.asarray(velocity, dtype=np.float64)
        speed_sq = np.einsum("ij,ij->i", velocity, velocity)
        total_pressure = (
            self._reference_pressure
            + 0.5 * self._density * float(np.dot(self._v_inf, self._v_inf))
            + self._pressure_jump_offsets(points)
        )
        return total_pressure - 0.5 * self._density * speed_sq

    def _pressure_jump_offsets(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return cumulative static-pressure offsets from actuator disks."""
        points = np.asarray(points, dtype=np.float64)
        offsets = np.zeros(points.shape[0], dtype=np.float64)
        for disk in self._disks:
            signed_distance, radial_distance, tolerance = self._disk_local_coordinates(points, disk)
            multiplier = np.where(
                signed_distance > tolerance,
                1.0,
                np.where(signed_distance < -tolerance, 0.0, 0.5),
            )
            active = radial_distance <= float(disk.config.radius) + tolerance
            multiplier = np.where(active, multiplier, 0.0)
            offsets += multiplier * float(disk.pressure_rise)
        return offsets

    def _regularize_disk_plane_velocity(
        self,
        points: NDArray[np.float64],
        velocity: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Replace on-disk-plane velocities with side-averaged samples."""
        corrected = np.asarray(velocity, dtype=np.float64).copy()
        for disk in self._disks:
            signed_distance, radial_distance, tolerance = self._disk_local_coordinates(points, disk)
            on_plane = (
                (np.abs(signed_distance) <= tolerance)
                & (radial_distance <= float(disk.config.radius) + tolerance)
            )
            if not np.any(on_plane):
                continue
            sample_offset = max(1e-6, 0.5 * float(np.sqrt(np.mean(disk.mesh.areas))))
            plus_points = points[on_plane] + sample_offset * disk.normal
            minus_points = points[on_plane] - sample_offset * disk.normal
            corrected[on_plane] = 0.5 * (
                self.velocity_at(plus_points) + self.velocity_at(minus_points)
            )
        return corrected

    def _disk_local_coordinates(
        self,
        points: NDArray[np.float64],
        disk: ActuatorDiskRuntime,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """Return signed plane distance, radial distance, and tolerance."""
        center = np.asarray(disk.config.center, dtype=np.float64)
        relative = np.asarray(points, dtype=np.float64) - center
        signed_distance = np.einsum("ij,j->i", relative, disk.normal)
        in_plane = relative - np.outer(signed_distance, disk.normal)
        radial_distance = np.linalg.norm(in_plane, axis=1)
        tolerance = max(1e-10, 1e-8 * float(disk.config.radius))
        return signed_distance, radial_distance, tolerance

    def _velocity_at_disk(self, disk: ActuatorDiskRuntime) -> NDArray[np.float64]:
        offset = disk.sample_offset * disk.normal
        plus_points = disk.mesh.centers + offset
        minus_points = disk.mesh.centers - offset

        velocity_plus = self._body_solver.velocity_at(plus_points)
        velocity_minus = self._body_solver.velocity_at(minus_points)
        for other in self._disks:
            velocity_plus += compute_point_doublet_velocity(
                plus_points,
                other.mesh,
                other.doublet_strength,
            )
            velocity_minus += compute_point_doublet_velocity(
                minus_points,
                other.mesh,
                other.doublet_strength,
            )
        for inlet in self._inlets:
            velocity_plus += compute_all_velocities_influence(plus_points, inlet.mesh.nodes, inlet.mesh.panels, inlet.source_strength)
            velocity_minus += compute_all_velocities_influence(minus_points, inlet.mesh.nodes, inlet.mesh.panels, inlet.source_strength)
        for outlet in self._outlets:
            velocity_plus += compute_all_velocities_influence(plus_points, outlet.mesh.nodes, outlet.mesh.panels, outlet.source_strength)
            velocity_minus += compute_all_velocities_influence(minus_points, outlet.mesh.nodes, outlet.mesh.panels, outlet.source_strength)
        return 0.5 * (velocity_plus + velocity_minus)

    def _write_outputs(self) -> None:
        adm_dir = self._case_dir / "out" / "adm"
        adm_dir.mkdir(parents=True, exist_ok=True)
        plot_adm_convergence(self._history, adm_dir / "adm_convergence.png")
        plot_fan_curve_progression(
            self._disks,
            self._history,
            adm_dir / "adm_fan_curve_progression.png",
        )
        save_adm_solver_run(
            path=self._case_dir / "out" / "solverRuns" / "adm_solution.npz",
            solver=self,
        )

        for index, disk in enumerate(self._disks):
            disk.mesh.cell_data["pressure_rise"] = np.full(disk.mesh.num_panels, disk.pressure_rise)
            disk.mesh.cell_data["doublet_strength"] = disk.doublet_strength
            disk.mesh.cell_data["normal_velocity"] = disk.normal_velocity
            disk.mesh.cell_data["panel_area"] = disk.mesh.areas
            disk.mesh.save_vtk(str(adm_dir / f"{index:02d}_{disk.config.name}_disk.vtp"))
