import numpy as np
from numpy.typing import NDArray

from ..base import ThermalSolver, ThermalResult
from ..utils import compute_total_heat_rate
from .discretization import (
    assemble_boundary_matrices,
    assemble_domain_matrices,
    assemble_boundary_domain_coupling
)

class BDIMThermalSolver(ThermalSolver):
    """
    Implements the full Boundary-Domain Integral Method for the thermal boundary layer
    governed by Gao et al. (2013). This solver directly computes surface distributions 
    by solving a fully coupled linear matrix system taking field mapping kinematics and
    boundary heat flux constraints as strict inputs.
    """
    
    def __init__(self,
                 fluid_properties: dict,          # dict containing 'rho', 'mu', 'k', 'cp'
                 T_inf: float,
                 # Boundary Arrays
                 arc_length: NDArray,
                 nodes_b: NDArray,                # (N, 2)
                 normals_b: NDArray,              # (N, 2)
                 lengths_b: NDArray,              # (N,)
                 u_b: NDArray,                    # (N, 2) velocity vector along boundary
                 # Domain Field Arrays
                 nodes_domain: NDArray,           # (K, 2)
                 areas_domain: NDArray,           # (K,)
                 u_domain: NDArray,               # (K, 2) velocity vector in domain
                 grad_u_domain: NDArray,          # (K, 2, 2) velocity gradient tensor matrix
                 p_domain: NDArray,               # (K,) explicit fluid pressure 
                 # BC
                 q_wall: NDArray = None,          # (N,) known heat flux
                 T_wall: NDArray = None           # (N,) known temperature
                 ):
        
        # We manually map the dict to the base solver properties requested in the interface
        self.rho = fluid_properties.get('rho', 1.225)
        self.mu = fluid_properties.get('mu', 1.81e-5)
        self.cp = fluid_properties.get('cp', 1005.0)
        
        # Build strict signature variables for base
        prandtl = (self.cp * self.mu) / fluid_properties.get('k', 0.026)
        
        super().__init__(
            bl_result=None, # Not explicitly required for this mode (using full domain instead)
            T_wall=T_wall,
            T_inf=T_inf,
            Pr=prandtl,
            k=fluid_properties.get('k', 0.026),
            q_wall=q_wall
        )
        
        self.arc_length = arc_length
        self.nodes_b = nodes_b
        self.normals_b = normals_b
        self.lengths_b = lengths_b
        self.u_b = u_b
        
        self.nodes_domain = nodes_domain
        self.areas_domain = areas_domain
        self.u_domain = u_domain
        self.grad_u_domain = grad_u_domain
        self.p_domain = p_domain

    def _compute_hydrodynamic_source_w(self) -> tuple[NDArray, NDArray]:
        """
        Calculates the internal domain mechanical/convective interaction vectors {w}.
        Decomposes directly into structural constant mappings {c_w} and boundary proportional elements.
        """
        K = len(self.nodes_domain)
        c_w_I = np.zeros((K, 2))
        
        for k in range(K):
            u_vec = self.u_domain[k]
            grad_u = self.grad_u_domain[k]  # [[dudx, dudy], [dvdx, dvdy]]
            p = self.p_domain[k]
            
            # Form shear strain rate tensor
            S = grad_u + grad_u.T
            viscous_term = self.mu * (S @ u_vec)
            
            # Form kinetic / pressure energy structural tensor
            kinetic_term = (p + 0.5 * self.rho * np.dot(u_vec, u_vec)) * u_vec
            c_w_I[k] = viscous_term - kinetic_term
            
        rho_cv_u_I = self.rho * self.cp * self.u_domain  # Specific heat capacity mapping
        
        return c_w_I, rho_cv_u_I

    def solve(self) -> ThermalResult:
        N = len(self.nodes_b)
        K = len(self.nodes_domain)
        
        # 1. Assemble foundational Green's Function integration kernels
        H, G = assemble_boundary_matrices(self.nodes_b, self.normals_b, self.lengths_b)
        
        E_b_dom = assemble_domain_matrices(self.nodes_b, self.nodes_domain, self.areas_domain)
        EC_b = assemble_boundary_domain_coupling(self.nodes_b, self.nodes_b, self.normals_b, self.lengths_b)
        
        E_I_dom = assemble_domain_matrices(self.nodes_domain, self.nodes_domain, self.areas_domain)
        EC_I = assemble_boundary_domain_coupling(self.nodes_domain, self.nodes_b, self.normals_b, self.lengths_b)
        
        # 2. Reconstruct internal mechanical dissipation mappings
        c_w_I, rho_cv_u_I = self._compute_hydrodynamic_source_w()
        
        c_w_b = np.zeros((N, 2)) # Assuming explicit gradient boundaries are neglected analytically
        rho_cv_u_b = self.rho * self.cp * self.u_b
        
        # Formulate decoupled dependencies separating {T_b} and {T_I} variables
        D_b = np.zeros((N, K))
        for i in range(N):
            for k in range(K):
                D_b[i, k] = np.dot(E_b_dom[i, k], rho_cv_u_I[k])
                
        D_I = np.zeros((K, K))
        for i in range(K):
            for k in range(K):
                if i != k:
                    D_I[i, k] = np.dot(E_I_dom[i, k], rho_cv_u_I[k])
                    
        B_b = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                B_b[i, j] = np.dot(EC_b[i, j], rho_cv_u_b[j])
                
        B_I = np.zeros((K, N))
        for i in range(K):
            for j in range(N):
                B_I[i, j] = np.dot(EC_I[i, j], rho_cv_u_b[j])
                
        # Knowns vector mapping
        const_b = np.sum(EC_b * c_w_b[:, None, :], axis=2).sum(axis=1) - np.sum(E_b_dom * c_w_I[:, None, :], axis=2).sum(axis=1)
        const_I = np.sum(EC_I * c_w_b[:, None, :], axis=2).sum(axis=1) - np.sum(E_I_dom * c_w_I[:, None, :], axis=2).sum(axis=1)
        
        # 3. Impose exact linear combinations constraints and execute algebraic inversion
        if self.q_wall is not None:
            # We must solve simultaneously for {T_b} (Boundary temperatures) and {T_I}
            from .kernels import temp_fundamental, temp_normal_derivative
            
            # Must approximate G_I and H_I dependencies
            G_I = np.zeros((K, N))
            H_I = np.zeros((K, N))
            for i in range(K):
                for j in range(N):
                    G_I[i, j] = -temp_fundamental(self.nodes_domain[i], self.nodes_b[j]) * self.lengths_b[j]
                    H_I[i, j] = temp_normal_derivative(self.nodes_domain[i], self.nodes_b[j], self.normals_b[j]) * self.lengths_b[j]

            # Enforce q_w uniformly if constant scalar is passed
            q_b = np.full(N, self.q_wall) if isinstance(self.q_wall, (int, float)) else self.q_wall
            
            Y_b = G @ q_b + const_b
            Y_I = G_I @ q_b + const_I
            
            # The structured full matrix system block
            Sys_mat = np.block([
                [H - B_b,             D_b],
                [H_I - B_I, np.eye(K) + D_I]
            ])
            
            Sys_rhs = np.concatenate([Y_b, Y_I])
            
            solution = np.linalg.solve(Sys_mat, Sys_rhs)
            T_b = solution[:N]
            
        else:
            raise NotImplementedError("Dirichlet (Known T_w, Unknown q_w) condition matrix resolution framework mapping not configured for simplified object structure.")
        
        # 4. Integrate finalized dimensional metrics maps: Nusselt, Heat Transfer Coef h(s)
        delta_T = T_b - self.T_inf
        h = np.divide(q_b, delta_T, out=np.zeros_like(q_b), where=(np.abs(delta_T) > 1e-10))
        
        characteristic_L = float(np.max(self.arc_length)) if len(self.arc_length) > 0 else 1.0
        nu_s = (h * characteristic_L) / self.k
        
        total_q = compute_total_heat_rate(q_b, self.arc_length)
        
        return ThermalResult(
            arc_length=self.arc_length,
            nusselt=nu_s,
            heat_transfer_coeff=h,
            wall_heat_flux=q_b,
            thermal_bl_thickness=np.zeros_like(self.arc_length), # Stably mocked (as pure object ignores secondary field metrics)
            total_heat_rate=total_q,
            wall_temperature=T_b
        )
