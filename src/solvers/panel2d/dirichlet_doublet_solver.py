"""
Constant-strength Dirichlet doublet panel method solver (Morino formulation).

Implements the combined source + doublet method with Dirichlet internal-potential
boundary condition from Katz & Plotkin, Sections 11.3.1 and 11.5.1.

Morino formulation:
    - Source strengths σ_j are *prescribed*: σ_j = n̂_j · V∞
    - Doublet strengths μ_j are *unknowns*, solved from the Dirichlet BC
      (zero perturbation potential inside the body): C·μ = −B·σ
    - Surface velocity: V_t = dμ/ds + V∞ · t̂  (K&P Eq. 11.76)

Bluff-body adaptation (non-lifting, no wake):
    - Wake panels deleted (μ_W = 0)
    - Uniform-doublet null-space cured by pinning μ₁ = 0 per component
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Tuple

from core.geometry.mesh import Mesh
from .base import PanelSolver2D, PanelMethodConfig
from .influences.doublet import (
    compute_doublet_influence_matrix,
    compute_source_potential_matrix,
    compute_doublet_velocity_influence,
)
from .influences.source import compute_source_velocity_influence


class DirichletDoubletSolver(PanelSolver2D):
    """
    2D Morino source + doublet solver with Dirichlet internal-potential BC.

    Adapted for non-lifting bluff bodies:
    - No wake panels (μ_W = 0).
    - Singular system cured by grounding one doublet node per component (μ₁ = 0).
    - Surface velocity from doublet-strength gradient: V_t = dμ/ds + V∞·t̂.

    Suitable for closed bodies in subsonic potential flow where lift is not required.
    """

    def __init__(self, mesh: Mesh, v_inf: float = 1.0, aoa: float = 0.0):
        super().__init__(mesh, v_inf, aoa)
        self._mu: NDArray[np.float64] | None = None
        self._sigma: NDArray[np.float64] | None = None

    @property
    def config(self) -> PanelMethodConfig:
        return PanelMethodConfig(
            singularity_type="source_doublet",
            panel_order="constant",
            panel_geometry="flat",
        )

    @property
    def mu(self) -> NDArray[np.float64]:
        """Doublet strengths for each panel."""
        if self._mu is None:
            raise RuntimeError("Solver not executed. Call solve() first.")
        return self._mu

    @property
    def sigma(self) -> NDArray[np.float64]:
        """Source strengths for each panel (prescribed: σ = n̂ · V∞)."""
        if self._sigma is None:
            raise RuntimeError("Solver not executed. Call solve() first.")
        return self._sigma

    @property
    def Vt(self) -> NDArray[np.float64]:
        """Tangential velocity magnitude at panel centres (N,)."""
        if not self.is_solved:
            raise RuntimeError("Solver not executed. Call solve() first.")
        return self.surface_velocity[:, 0]

    @property
    def Cp(self) -> NDArray[np.float64]:
        """Pressure coefficient via Bernoulli: Cp = 1 − (Vt / V∞)²."""
        if not self.is_solved:
            raise RuntimeError("Solver not executed. Call solve() first.")
        return 1.0 - (self.Vt / self._v_inf) ** 2

    # ── Abstract method implementations ─────────────────────────────────

    def _compute_influence_matrices(self) -> Dict[str, NDArray]:
        """Build doublet (C) and source-potential (B) influence matrices."""
        C = compute_doublet_influence_matrix(self._mesh)
        B = compute_source_potential_matrix(self._mesh)
        return {"C": C, "B": B}

    def _solve_linear_system(
        self, influence_matrices: Dict[str, NDArray]
    ) -> Dict[str, NDArray]:
        """
        Solve Morino system: C·μ = −B·σ with σ_j = n̂_j · V∞ prescribed.

        A uniform doublet distribution lies in the null-space of C for a
        closed body (it corresponds to a constant interior potential shift).
        We remove this degree of freedom by pinning the first panel of each
        component to μ = 0.
        """
        C = influence_matrices["C"]
        B = influence_matrices["B"]
        n_panels = self._mesh.num_panels

        # Prescribed source strengths: σ_j = n̂_j · V∞
        normals = self._mesh.normals[:, :2]  # (N, 2)
        v_inf_2d = self.v_inf_vector[:2]  # (2,)
        sigma = normals @ v_inf_2d  # (N,)

        # Right-hand side: −B·σ
        rhs = -B @ sigma

        # Pin μ = 0 for the first panel of each component to cure singularity
        C_pinned = C.copy()
        rhs_pinned = rhs.copy()

        unique_comps = np.unique(self._mesh.component_ids)
        for comp_id in unique_comps:
            comp_panels = np.where(self._mesh.component_ids == comp_id)[0]
            pin_idx = comp_panels[0]
            C_pinned[pin_idx, :] = 0.0
            C_pinned[pin_idx, pin_idx] = 1.0
            rhs_pinned[pin_idx] = 0.0

        # Solve
        try:
            mu = np.linalg.solve(C_pinned, rhs_pinned)
        except np.linalg.LinAlgError:
            raise RuntimeError(
                "Singular matrix in DirichletDoubletSolver. "
                "Check geometry or component connectivity."
            )

        self._mu = mu
        self._sigma = sigma

        # Store in mesh for backward compatibility
        self._mesh.cell_data["doublet_strength"] = mu
        self._mesh.cell_data["source_strength"] = sigma

        return {"doublet": mu, "source": sigma}

    def _compute_surface_velocity(
        self,
        influence_matrices: Dict[str, NDArray],
        strengths: Dict[str, NDArray],
    ) -> NDArray[np.float64]:
        """
        Compute surface tangential velocity from doublet-strength gradient.

        K&P Eq. 11.76:  V_t = dμ/ds + V∞ · t̂

        The derivative is evaluated per-component using central differences
        with periodic wrap-around for each closed body.
        """
        mu = strengths["doublet"]
        n_panels = self._mesh.num_panels
        centers = self._mesh.centers[:, :2]
        tangents = self._mesh.tangents[:, :2]
        v_inf_2d = self.v_inf_vector[:2]

        # Freestream tangential component at each panel
        Vt_freestream = tangents @ v_inf_2d

        # Differentiate μ along arc length per component
        dmu_ds = np.zeros(n_panels, dtype=np.float64)
        unique_comps = np.unique(self._mesh.component_ids)

        for comp_id in unique_comps:
            idx = np.where(self._mesh.component_ids == comp_id)[0]
            n_comp = len(idx)

            if n_comp < 2:
                continue

            # Arc lengths at panel centres within this component
            comp_centers = centers[idx]
            arc = np.zeros(n_comp)
            for k in range(1, n_comp):
                arc[k] = arc[k - 1] + np.linalg.norm(
                    comp_centers[k] - comp_centers[k - 1]
                )
            # Closure distance (last → first)
            total_arc = arc[-1] + np.linalg.norm(
                comp_centers[0] - comp_centers[-1]
            )

            comp_mu = mu[idx]

            # Central differences with periodic wrap-around
            for k in range(n_comp):
                k_plus = (k + 1) % n_comp
                k_minus = (k - 1) % n_comp

                if k == 0:
                    ds_minus = arc[0] + (total_arc - arc[-1])
                else:
                    ds_minus = arc[k] - arc[k_minus]

                if k == n_comp - 1:
                    ds_plus = total_arc - arc[k]
                else:
                    ds_plus = arc[k_plus] - arc[k]

                ds_total = ds_minus + ds_plus
                if ds_total < 1e-14:
                    continue

                dmu_ds[idx[k]] = (comp_mu[k_plus] - comp_mu[k_minus]) / ds_total

        Vt = dmu_ds + Vt_freestream
        return Vt

    def _velocity_at_points(
        self, points: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute velocity at arbitrary field points from doublet + source panels.

        The total velocity at a field point is:
            V = V∞ + ∑_j μ_j · V_doublet_j + ∑_j σ_j · V_source_j / (2π)
        """
        if self._mu is None or self._sigma is None:
            raise RuntimeError("Must solve before evaluating velocity field.")

        n_panels = self._mesh.num_panels
        panel_start_idx = self._mesh.panels[:, 0]
        panel_end_idx = self._mesh.panels[:, 1]
        nodes_start = self._mesh.nodes[panel_start_idx, :2]
        nodes_end = self._mesh.nodes[panel_end_idx, :2]
        panel_lengths = self._mesh.areas
        tangents_2d = self._mesh.tangents[:, :2]
        phi = np.arctan2(tangents_2d[:, 1], tangents_2d[:, 0])
        phi = np.where(phi < 0, phi + 2 * np.pi, phi)

        num_points = len(points)
        Vx = np.zeros(num_points, dtype=np.float64)
        Vy = np.zeros(num_points, dtype=np.float64)

        for idx in range(num_points):
            pt = points[idx, :2]

            for j in range(n_panels):
                # Doublet contribution
                du_d, dw_d = compute_doublet_velocity_influence(
                    point=pt,
                    panel_start=nodes_start[j],
                    panel_end=nodes_end[j],
                )
                Vx[idx] += self._mu[j] * du_d
                Vy[idx] += self._mu[j] * dw_d

                # Source contribution
                Mx, My = compute_source_velocity_influence(
                    point=pt,
                    panel_start=nodes_start[j],
                    panel_length=panel_lengths[j],
                    panel_angle=phi[j],
                )
                Vx[idx] += self._sigma[j] * Mx / (2.0 * np.pi)
                Vy[idx] += self._sigma[j] * My / (2.0 * np.pi)

        # Add freestream
        Vx += self.v_inf_vector[0]
        Vy += self.v_inf_vector[1]

        return Vx, Vy

    # ── Validation helpers ──────────────────────────────────────────────

    def _compute_induced_normal_velocity(self) -> NDArray[np.float64]:
        """Compute induced normal velocity (for BC verification)."""
        # For Dirichlet formulation, check internal potential = 0 instead
        C = self._influence_matrices["C"]
        B = self._influence_matrices["B"]
        phi_int = C @ self._mu + B @ self._sigma
        return phi_int  # Should be ≈ 0 everywhere

    def _perform_solver_specific_validation(self):
        """Validate doublet solution: check interior potential ≈ 0."""
        phi_int = self._compute_induced_normal_velocity()
        max_err = np.max(np.abs(phi_int))
        rms_err = np.sqrt(np.mean(phi_int**2))
        print(f"\n--- Dirichlet Doublet Validation ---")
        print(f"  Interior potential max|Φ|:  {max_err:.2e}")
        print(f"  Interior potential RMS(Φ):  {rms_err:.2e}")

        mu_sum = np.sum(self._mu)
        print(f"  Doublet strength Σμ:        {mu_sum:.6f}")
        print(f"  Doublet range:              [{self._mu.min():.6f}, {self._mu.max():.6f}]")
        print(f"  Source range:               [{self._sigma.min():.6f}, {self._sigma.max():.6f}]")
