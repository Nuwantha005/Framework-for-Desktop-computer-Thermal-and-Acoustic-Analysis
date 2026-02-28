"""
Linear-strength source/doublet panel method solver (Dirichlet formulation).

Implements the higher-order Morino method from K&P §11.5.1 with linearly
varying source and doublet distributions on flat panels.

Key improvements over the constant Dirichlet doublet solver:
- Linear doublet/source distributions → continuous singularity strengths at nodes
- Surface velocity via analytical per-panel derivative: V_t = (μ_j − μ_{j+1})/S_j + V∞·t̂
  (K&P Eq. 11.122) — no numerical central differences needed
- Node-based unknowns (N nodes for N panels on a closed body)

Formulation:
    - Source strengths σ_k at nodes are prescribed from averaged normals at nodes
    - Doublet strengths μ_k at nodes are unknowns, solved from Dirichlet BC
      (zero perturbation potential inside): C·μ = −B·σ
    - Bluff-body adaptation: no wake (μ_W = 0), pin μ₁ = 0 per component
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Tuple

from core.geometry.mesh import Mesh
from .base import PanelSolver2D, PanelMethodConfig
from .influences.linear_doublet import (
    compute_linear_doublet_influence_matrix,
    compute_linear_source_potential_matrix,
    compute_linear_doublet_velocity_field,
)
from .influences.linear_source import compute_linear_source_velocity_field


class LinearSourceDoubletSolver(PanelSolver2D):
    """
    2D linear-strength source/doublet solver with Dirichlet internal-potential BC.

    Higher-order Morino formulation adapted for non-lifting bluff bodies:
    - No wake panels (μ_W = 0).
    - Gauge fix: pin μ₁ = 0 per component.
    - Surface velocity: V_t = (μ_j − μ_{j+1})/S_j + V∞·t̂ (analytical derivative).

    Node-based unknowns for both source and doublet. For closed bodies,
    N_nodes == N_panels.
    """

    def __init__(self, mesh: Mesh, v_inf: float = 1.0, aoa: float = 0.0):
        super().__init__(mesh, v_inf, aoa)
        self._mu: NDArray[np.float64] | None = None
        self._sigma: NDArray[np.float64] | None = None

    @property
    def config(self) -> PanelMethodConfig:
        return PanelMethodConfig(
            singularity_type="source_doublet",
            panel_order="linear",
            panel_geometry="flat",
        )

    @property
    def mu(self) -> NDArray[np.float64]:
        """Doublet strengths at nodes."""
        if self._mu is None:
            raise RuntimeError("Solver not executed. Call solve() first.")
        return self._mu

    @property
    def sigma(self) -> NDArray[np.float64]:
        """Source strengths at nodes (prescribed from geometry + freestream)."""
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
        """Build linear doublet (C) and linear source potential (B) matrices."""
        C = compute_linear_doublet_influence_matrix(self._mesh)
        B = compute_linear_source_potential_matrix(self._mesh)
        return {"C": C, "B": B}

    def _solve_linear_system(
        self, influence_matrices: Dict[str, NDArray]
    ) -> Dict[str, NDArray]:
        """
        Solve Dirichlet system: C·μ = −B·σ with node-based σ prescribed.

        Source strengths at nodes are computed as the average of the
        freestream normal component on adjacent panels (K&P §11.5.1).
        Gauge fix: pin μ₁ = 0 per component.
        """
        C = influence_matrices["C"]
        B = influence_matrices["B"]
        n_nodes = self._mesh.num_nodes

        # Prescribe source strengths at nodes: average of adjacent panel normals · V∞
        sigma = self._compute_node_source_strengths()

        # RHS: −B · σ
        rhs = -B @ sigma

        # Pin μ = 0 for the first node of each component (gauge fix)
        # Use lstsq for robustness — handles rank deficiency > 1 on symmetric meshes
        C_pinned = C.copy()
        rhs_pinned = rhs.copy()

        unique_comps = np.unique(self._mesh.component_ids)
        for comp_id in unique_comps:
            comp_panels = np.where(self._mesh.component_ids == comp_id)[0]
            # First node of first panel in this component
            pin_node = self._mesh.panels[comp_panels[0], 0]
            C_pinned[pin_node, :] = 0.0
            C_pinned[pin_node, pin_node] = 1.0
            rhs_pinned[pin_node] = 0.0

        # Solve using lstsq for robustness against extra null modes
        # (e.g., alternating mode on highly symmetric circular meshes)
        mu, residuals, rank, sv = np.linalg.lstsq(C_pinned, rhs_pinned, rcond=None)
        # Enforce gauge: shift so pinned node is exactly 0
        for comp_id in unique_comps:
            comp_panels = np.where(self._mesh.component_ids == comp_id)[0]
            pin_node = self._mesh.panels[comp_panels[0], 0]
            mu -= mu[pin_node]  # remove any residual constant offset

        self._mu = mu
        self._sigma = sigma

        self._mesh.cell_data["doublet_strength"] = mu
        self._mesh.cell_data["source_strength"] = sigma

        return {"doublet": mu, "source": sigma}

    def _compute_surface_velocity(
        self,
        influence_matrices: Dict[str, NDArray],
        strengths: Dict[str, NDArray],
    ) -> NDArray[np.float64]:
        """
        Compute surface tangential velocity from the doublet strength gradient.

        The exterior perturbation potential on the surface equals μ
        (from the Dirichlet BC: Φ_int = 0, Φ_ext = Φ_int + μ = μ).
        Therefore:
            V_t = V∞ · t̂  +  ∂μ/∂s

        Uses per-component central differences with periodic wrap-around
        (same scheme as the constant Dirichlet doublet solver) to achieve
        better accuracy than per-panel forward differences.
        """
        mu = strengths["doublet"]
        n_panels = self._mesh.num_panels
        centers = self._mesh.centers[:, :2]
        tangents = self._mesh.tangents[:, :2]
        v_inf_2d = self.v_inf_vector[:2]

        # Freestream tangential component at each panel
        Vt_freestream = tangents @ v_inf_2d

        # Differentiate μ along arc length per component using central diffs.
        # μ is defined at nodes; interpolate to panel centres first:
        #   μ_panel_j ≈ (μ_{n1} + μ_{n2}) / 2
        mu_panel = np.zeros(n_panels, dtype=np.float64)
        for j in range(n_panels):
            n1 = self._mesh.panels[j, 0]
            n2 = self._mesh.panels[j, 1]
            mu_panel[j] = 0.5 * (mu[n1] + mu[n2])

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

            comp_mu = mu_panel[idx]

            # Central differences with periodic wrap-around
            for k in range(n_comp):
                k_plus = (k + 1) % n_comp
                k_minus = (k - 1) % n_comp

                if k == 0:
                    ds_minus = arc[0] + (total_arc - arc[-1])
                else:
                    ds_minus = arc[k] - arc[k_minus]

                if k == n_comp - 1:
                    ds_plus = total_arc - arc[-1]
                else:
                    ds_plus = arc[k_plus] - arc[k]

                ds = ds_minus + ds_plus
                if ds > 1e-14:
                    dmu_ds[idx[k]] = (comp_mu[k_plus] - comp_mu[k_minus]) / ds

        Vt = dmu_ds + Vt_freestream
        return Vt

    def _velocity_at_points(
        self, points: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Velocity at arbitrary field points from linear doublet + source panels.

        V = V∞ + V_doublet(μ) + V_source(σ)
        """
        if self._mu is None or self._sigma is None:
            raise RuntimeError("Must solve before evaluating velocity field.")

        # Doublet contribution
        V_doublet = compute_linear_doublet_velocity_field(
            points, self._mesh, self._mu
        )

        # Source contribution
        V_source = compute_linear_source_velocity_field(
            points, self._mesh, self._sigma
        )

        Vx = self.v_inf_vector[0] + V_doublet[:, 0] + V_source[:, 0]
        Vy = self.v_inf_vector[1] + V_doublet[:, 1] + V_source[:, 1]

        return Vx, Vy

    # ── Helper methods ──────────────────────────────────────────────────

    def _compute_node_source_strengths(self) -> NDArray[np.float64]:
        """
        Prescribe source strengths at nodes from averaged panel normals.

        For each node, σ_k = n̂_avg_k · V∞ where n̂_avg is the average
        outward normal of the panels adjacent to node k.
        """
        n_nodes = self._mesh.num_nodes
        normals = self._mesh.normals[:, :2]
        v_inf_2d = self.v_inf_vector[:2]

        # Accumulate normal vectors at each node from adjacent panels
        node_normal = np.zeros((n_nodes, 2), dtype=np.float64)
        node_count = np.zeros(n_nodes, dtype=np.float64)

        for j in range(self._mesh.num_panels):
            n1 = self._mesh.panels[j, 0]
            n2 = self._mesh.panels[j, 1]
            nj = normals[j]
            node_normal[n1] += nj
            node_count[n1] += 1.0
            node_normal[n2] += nj
            node_count[n2] += 1.0

        # Average and normalize
        for k in range(n_nodes):
            if node_count[k] > 0:
                node_normal[k] /= node_count[k]
                norm = np.linalg.norm(node_normal[k])
                if norm > 1e-14:
                    node_normal[k] /= norm

        # σ_k = n̂_k · V∞
        sigma = node_normal @ v_inf_2d
        return sigma

    # ── Validation helpers ──────────────────────────────────────────────

    def _compute_induced_normal_velocity(self) -> NDArray[np.float64]:
        """Check interior potential (should be ≈ 0 for Dirichlet BC)."""
        C = self._influence_matrices["C"]
        B = self._influence_matrices["B"]
        return C @ self._mu + B @ self._sigma

    def _perform_solver_specific_validation(self):
        """Validate: check interior potential ≈ 0, print doublet statistics."""
        phi_int = self._compute_induced_normal_velocity()
        max_err = np.max(np.abs(phi_int))
        rms_err = np.sqrt(np.mean(phi_int**2))
        print(f"\n--- Linear Source/Doublet Validation ---")
        print(f"  Interior potential max|Φ|:  {max_err:.2e}")
        print(f"  Interior potential RMS(Φ):  {rms_err:.2e}")
        print(f"  Doublet range:              [{self._mu.min():.6f}, {self._mu.max():.6f}]")
        print(f"  Source range:               [{self._sigma.min():.6f}, {self._sigma.max():.6f}]")
