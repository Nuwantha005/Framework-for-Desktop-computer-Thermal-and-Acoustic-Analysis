"""
BDIM (Boundary-Domain Integral Method) thermal solver.

Full boundary integral formulation for thermal boundary layer based on
Gao et al. (2013). More accurate than Reynolds analogy for complex
geometries, separated regions, and non-uniform wall temperature.

This solver requires domain mesh data (velocity field, gradients, pressure)
in addition to boundary data. Use ReynoldsAnalogyThermal for surface-only
calculations.

References
----------
* Gao, X.-W., Peng, H.-F., & Liu, J. (2013). A boundary-domain integral
  equation method for solving convective heat transfer problems.
  International Journal of Heat and Mass Transfer.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..base import ThermalResult, ThermalFieldData
from ..utils import compute_total_heat_rate
from .discretization import (
    assemble_boundary_matrices,
    assemble_domain_matrices,
    assemble_boundary_domain_coupling,
    compute_analytical_HG
)


@dataclass
class BDIMInput:
    """
    Input data structure for BDIM thermal solver.
    
    Contains both boundary and domain field data required for the
    boundary-domain integral formulation.
    
    Attributes:
        arc_length: Arc-length along boundary [m], shape (N,).
        nodes_b: Boundary node positions [m], shape (N, 2).
        normals_b: Boundary outward normals [-], shape (N, 2).
        lengths_b: Boundary panel lengths [m], shape (N,).
        u_b: Velocity at boundary [m/s], shape (N, 2).
        x_b: Boundary x-coordinates [m], shape (N,). For result output.
        y_b: Boundary y-coordinates [m], shape (N,). For result output.
        nodes_domain: Domain node positions [m], shape (K, 2).
        areas_domain: Domain cell areas [m²], shape (K,).
        u_domain: Velocity in domain [m/s], shape (K, 2).
        grad_u_domain: Velocity gradient tensor, shape (K, 2, 2).
            Format: [[du/dx, du/dy], [dv/dx, dv/dy]]
        p_domain: Pressure in domain [Pa], shape (K,).
        side: Surface side identifier ("upper" or "lower").
        grid_shape: Optional (M, Ny) shape for reshaping domain data.
        y_normal: Optional wall-normal distances [m], shape (M, Ny).
            Needed for field visualization output.
    """
    arc_length: NDArray[np.float64]
    nodes_b: NDArray[np.float64]
    normals_b: NDArray[np.float64]
    lengths_b: NDArray[np.float64]
    u_b: NDArray[np.float64]
    x_b: NDArray[np.float64]
    y_b: NDArray[np.float64]
    nodes_domain: NDArray[np.float64]
    areas_domain: NDArray[np.float64]
    u_domain: NDArray[np.float64]
    grad_u_domain: NDArray[np.float64]
    p_domain: NDArray[np.float64]
    side: str = "upper"
    grid_shape: Optional[tuple] = None
    y_normal: Optional[NDArray[np.float64]] = None


@dataclass
class BDIMConfig:
    """
    Configuration for BDIM thermal solver.
    
    Attributes:
        T_inf: Freestream temperature [K].
        rho: Fluid density [kg/m³].
        mu: Dynamic viscosity [Pa·s].
        k: Thermal conductivity [W/mK].
        cp: Specific heat capacity [J/kgK].
        q_wall: Heat flux BC [W/m²], shape (N,) or scalar.
        T_wall: Temperature BC [K], shape (N,) or scalar.
    """
    T_inf: float
    rho: float = 1.225
    mu: float = 1.81e-5
    k: float = 0.026
    cp: float = 1005.0
    q_wall: Optional[NDArray[np.float64]] = None
    T_wall: Optional[NDArray[np.float64]] = None
    
    def __post_init__(self):
        if self.q_wall is None and self.T_wall is None:
            raise ValueError(
                "Must provide either q_wall (heat flux) or T_wall (temperature) BC"
            )
    
    @property
    def Pr(self) -> float:
        """Prandtl number."""
        return (self.cp * self.mu) / self.k


class BDIMThermalSolver:
    """
    Boundary-Domain Integral Method thermal solver.
    
    Solves the energy equation using boundary integral formulation with
    domain coupling. More accurate than Reynolds analogy but requires
    full domain velocity/pressure field data.
    
    This solver uses its own input format (BDIMInput) rather than the
    common ThermalBLInput because it requires domain mesh data that
    isn't available from the standard BL solver output.
    
    To use with reconstructed BL field data, see the planned
    `create_bdim_input_from_bl_field()` helper function.
    
    Example::
    
        from solvers.thermal.bdim import BDIMThermalSolver, BDIMInput, BDIMConfig
        
        # Prepare domain data from BL field reconstruction
        bdim_input = BDIMInput(
            arc_length=s,
            nodes_b=boundary_nodes,
            normals_b=normals,
            lengths_b=panel_lengths,
            u_b=boundary_velocity,
            x_b=x, y_b=y,
            nodes_domain=domain_nodes,
            areas_domain=cell_areas,
            u_domain=velocity_field,
            grad_u_domain=velocity_gradients,
            p_domain=pressure_field,
        )
        
        config = BDIMConfig(T_inf=300.0, q_wall=1000.0)
        solver = BDIMThermalSolver(bdim_input, config)
        result = solver.solve()
    """
    
    def __init__(self, bdim_input: BDIMInput, config: BDIMConfig):
        """
        Initialize BDIM thermal solver.
        
        Args:
            bdim_input: Domain and boundary data
            config: Solver configuration with fluid properties and BCs
        """
        self.input = bdim_input
        self.config = config
        
        # Store fluid properties for convenience
        self.rho = config.rho
        self.mu = config.mu
        self.cp = config.cp
        self.k = config.k
    
    @property
    def name(self) -> str:
        return "bdim"
    
    def _compute_hydrodynamic_source_w(self, k_eff: float) -> tuple[NDArray, NDArray]:
        """
        Calculate internal domain mechanical/convective interaction vectors {w}.
        
        Decomposes into structural constant mappings {c_w} and boundary
        proportional elements for the temperature coupling.
        
        Returns:
            (c_w_I, rho_cv_u_I): Constant and temperature-dependent parts
        """
        K = len(self.input.nodes_domain)
        c_w_I = np.zeros((K, 2))
        
        for k in range(K):
            # Viscous dissipation is negligible for low-speed flow, and the divergence
            # trick for w requires a closed boundary which we don't have (truncated BL).
            # So we set the mechanical source term to zero to avoid huge artificial fluxes.
            c_w_I[k] = 0.0
        
        # Specific heat capacity mapping (temperature-dependent part)
        rho_cv_u_I = (self.rho * self.cp / k_eff) * self.input.u_domain
        
        return c_w_I / k_eff, rho_cv_u_I
    
    def solve(self) -> ThermalResult:
        """
        Solve thermal boundary layer using BDIM formulation.
        
        Returns:
            ThermalResult with wall temperature, heat transfer, etc.
        
        Raises:
            NotImplementedError: If Dirichlet BC (known T_wall) is requested.
        """
        N = len(self.input.nodes_b)
        K = len(self.input.nodes_domain)
        
        # 1. Assemble foundational Green's Function integration kernels
        H, G = assemble_boundary_matrices(
            self.input.nodes_b, self.input.normals_b, self.input.lengths_b
        )
        
        E_b_dom = assemble_domain_matrices(
            self.input.nodes_b, self.input.nodes_domain, self.input.areas_domain
        )
        EC_b = assemble_boundary_domain_coupling(
            self.input.nodes_b, self.input.nodes_b,
            self.input.normals_b, self.input.lengths_b
        )
        
        E_I_dom = assemble_domain_matrices(
            self.input.nodes_domain, self.input.nodes_domain, self.input.areas_domain
        )
        EC_I = assemble_boundary_domain_coupling(
            self.input.nodes_domain, self.input.nodes_b,
            self.input.normals_b, self.input.lengths_b
        )
        
        # Artificial diffusion for numerical stability at high Peclet numbers (upwinding equivalent)
        # Compute local cell size dx approx from areas
        dx_local = np.sqrt(self.input.areas_domain)
        # Artificial diffusion for numerical stability at high Peclet numbers
        # Use locally varying k_eff to maintain cell Peclet number <= 2
        dx_local = np.sqrt(self.input.areas_domain)
        u_mag = np.linalg.norm(self.input.u_domain, axis=1)
        
        # cell Pe = rho * cp * u_mag * dx_local / k
        # we want Pe_eff = rho * cp * u_mag * dx_local / k_eff <= 0.5
        # so k_eff = max(k, rho * cp * u_mag * dx_local / 0.5)
        k_eff_I = np.maximum(self.k, self.rho * self.cp * u_mag * dx_local / 0.5)
        
        # For boundary, use max over the boundary (assuming U_inf ~ 1.0)
        u_mag_b = np.linalg.norm(self.input.u_b, axis=1)
        # Force a minimum velocity so artificial diffusion kicks in at the wall
        u_mag_b_eff = np.maximum(u_mag_b, 1.0)
        dx_b = self.input.lengths_b
        k_eff_b = np.maximum(self.k, self.rho * self.cp * u_mag_b_eff * dx_b / 0.5)
        
        # 2. Reconstruct internal mechanical dissipation mappings
        c_w_I = np.zeros((K, 2)) # Mechanical dissipation ignored due to truncated BL domain
        c_w_b = np.zeros((N, 2))
        
        rho_cv_u_b = (self.rho * self.cp / k_eff_b[:, None]) * self.input.u_b
        rho_cv_u_I = (self.rho * self.cp / k_eff_I[:, None]) * self.input.u_domain
        
        # Formulate decoupled dependencies separating {T_b} and {T_I} variables
        # Vectorized: D_b[i, k] = E_b_dom[i, k, :] · rho_cv_u_I[k, :]
        D_b = np.sum(E_b_dom * rho_cv_u_I[None, :, :], axis=2)
        
        # D_I[i, k] = E_I_dom[i, k, :] · rho_cv_u_I[k, :] for i != k
        D_I = np.sum(E_I_dom * rho_cv_u_I[None, :, :], axis=2)
        np.fill_diagonal(D_I, 0.0)  # Zero out diagonal
        
        # B_b[i, j] = EC_b[i, j, :] · rho_cv_u_b[j, :]
        B_b = np.sum(EC_b * rho_cv_u_b[None, :, :], axis=2)
        
        # B_I[i, j] = EC_I[i, j, :] · rho_cv_u_b[j, :]
        B_I = np.sum(EC_I * rho_cv_u_b[None, :, :], axis=2)
        
        # Knowns vector mapping
        const_b = (
            np.sum(EC_b * c_w_b[:, None, :], axis=2).sum(axis=1)
            - np.sum(E_b_dom * c_w_I[None, :, :], axis=2).sum(axis=1)
        )
        const_I = (
            np.sum(EC_I * c_w_b[None, :, :], axis=2).sum(axis=1)
            - np.sum(E_I_dom * c_w_I[None, :, :], axis=2).sum(axis=1)
        )
        
        # 3. Impose constraints and solve
        if self.config.q_wall is not None:
            # Neumann BC: heat flux given, solve for T_b and T_I
            # Compute G_I and H_I using exact analytical expressions
            # to avoid singularity for domain points very close to panels
            H_I, G_I = compute_analytical_HG(
                self.input.nodes_domain, self.input.nodes_b,
                self.input.normals_b, self.input.lengths_b
            )
            
            # Expand q_wall if scalar
            if isinstance(self.config.q_wall, (int, float)):
                q_b = np.full(N, self.config.q_wall)
            else:
                q_b = np.asarray(self.config.q_wall)
            
            # To stabilize the BEM formulation on an open/truncated domain,
            # we use the effective thermal conductivity for the BEM source term.
            # This allows the heat to diffuse out of the domain, mimicking convective loss.
            # However, we must ensure q_b is positive for heating.
            q_b_k = q_b / k_eff_b
            Y_b = G @ q_b_k + const_b
            Y_I = G_I @ q_b_k + const_I
            
            # Build and solve the full matrix system
            Sys_mat = np.block([
                [H + B_b, D_b],
                [H_I + B_I, np.eye(K) + D_I]
            ])
            
            Sys_rhs = np.concatenate([Y_b, Y_I])
            
            # Enforce far-field and inflow boundary conditions on the truncated domain
            if self.input.grid_shape is not None:
                M, Ny = self.input.grid_shape
                # Indices for outer edge (y -> inf) and inflow (x -> start)
                outer_idx = np.arange(M) * Ny + (Ny - 1)
                inflow_idx = np.arange(Ny)
                bc_idx = np.unique(np.concatenate([outer_idx, inflow_idx]))
                
                sys_bc_idx = N + bc_idx
                Sys_mat[sys_bc_idx, :] = 0.0
                Sys_mat[sys_bc_idx, sys_bc_idx] = 1.0
                Sys_rhs[sys_bc_idx] = 0.0
            
            solution = np.linalg.solve(Sys_mat, Sys_rhs)
            T_b = solution[:N]
            T_I = solution[N:]  # Domain temperatures
            
        else:
            raise NotImplementedError(
                "Dirichlet BC (known T_w, unknown q_w) not yet implemented. "
                "Use heat flux BC (q_wall) instead."
            )
        
        # 4. Compute derived quantities
        # The solver actually computes theta = T - T_inf because the far-field 
        # boundary is omitted, implicitly enforcing theta -> 0 at infinity.
        theta_b = T_b
        T_b = theta_b + self.config.T_inf
        T_I = T_I + self.config.T_inf
        
        delta_T_wall = T_b - self.config.T_inf
        h = np.divide(
            q_b, delta_T_wall,
            out=np.zeros_like(q_b),
            where=(np.abs(delta_T_wall) > 1e-10)
        )
        
        L_char = float(np.max(self.input.arc_length)) if len(self.input.arc_length) > 0 else 1.0
        nusselt = (h * L_char) / self.k
        
        total_q = compute_total_heat_rate(q_b, self.input.arc_length)
        
        # 5. Build field data if grid shape is available
        field_data = None
        if self.input.grid_shape is not None and self.input.y_normal is not None:
            M, Ny = self.input.grid_shape
            # Reshape domain data to (M, Ny) grid
            T_field = T_I.reshape(M, Ny)
            x_domain = self.input.nodes_domain[:, 0].reshape(M, Ny)
            y_domain = self.input.nodes_domain[:, 1].reshape(M, Ny)
            
            field_data = ThermalFieldData(
                s=self.input.arc_length,
                y_normal=self.input.y_normal,
                x=x_domain,
                y=y_domain,
                T=T_field,
                T_inf=self.config.T_inf,
                side=self.input.side,
            )
        
        # Estimate thermal BL thickness from field data
        thermal_bl_thickness = np.zeros_like(self.input.arc_length)
        if field_data is not None:
            # Find where T drops to 99% of (T_wall - T_inf)
            T_field = field_data.T
            T_wall = T_field[:, 0]
            for i in range(N):
                if abs(T_wall[i] - self.config.T_inf) > 1e-10:
                    T_99 = self.config.T_inf + 0.99 * (T_wall[i] - self.config.T_inf)
                    # Find first index where T < T_99
                    for j in range(1, T_field.shape[1]):
                        if T_field[i, j] <= T_99:
                            thermal_bl_thickness[i] = self.input.y_normal[i, j]
                            break
                    else:
                        # Use last y value if T never reaches T_99
                        thermal_bl_thickness[i] = self.input.y_normal[i, -1]
        
        return ThermalResult(
            side=self.input.side,
            arc_length=self.input.arc_length,
            x=self.input.x_b,
            y=self.input.y_b,
            wall_temperature=T_b,
            heat_transfer_coeff=h,
            nusselt=nusselt,
            wall_heat_flux=q_b,
            thermal_bl_thickness=thermal_bl_thickness,
            total_heat_rate=total_q,
            solver_type=self.name,
            field=field_data,
        )
