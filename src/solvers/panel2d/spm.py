
import numpy as np
import math
from typing import Tuple, Optional
from core.geometry.mesh import Mesh

class SourcePanelSolver:
    """
    2D Constant Source Panel Method Solver.
    Assumes pure potential flow (no lift, no wake).
    Based on formulation in 'Low Speed Aerodynamics', Katz & Plotkin.
    """
    
    def __init__(self, mesh: Mesh, v_inf: float = 1.0, aoa: float = 0.0):
        """
        Initialize the solver.
        
        Args:
            mesh: The discretized geometry mesh (must be 2D).
            v_inf: Freestream velocity magnitude.
            aoa: Angle of attack in degrees.
        """
        if mesh.dimension != 2:
            raise ValueError("SourcePanelSolver only supports 2D meshes.")
            
        self.mesh = mesh
        self.v_inf = v_inf
        self.aoa = np.radians(aoa)
        
        # Results
        self.sigma = None # Source strengths
        self.Vt = None    # Tangential velocities
        self.Cp = None    # Pressure coefficients
        
    def compute_geometric_integrals(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the normal (I) and tangential (J) influence coefficient matrices.
        
        Returns:
            I: (N, N) matrix of normal velocity influence coefficients.
            J: (N, N) matrix of tangential velocity influence coefficients.
        """
        n_panels = len(self.mesh.panels)
        
        # Extract geometry components
        # Control points (N, 3) -> (N, 2)
        XC = self.mesh.centers[:, 0]
        YC = self.mesh.centers[:, 1]
        
        # Panel endpoints. For 2D lines, panels are (N, 2) indices.
        # We need the starting point of each panel (xb, yb) for the formula.
        # Assuming CCW or ordered panels, node 0 of panel j is the start.
        node_indices = self.mesh.panels[:, 0]
        XB = self.mesh.nodes[node_indices, 0]
        YB = self.mesh.nodes[node_indices, 1]
        
        # Panel lengths
        S = self.mesh.areas
        
        # Panel orientation angles (phi)
        # Tangents are normalized vectors (tx, ty). phi = atan2(ty, tx)
        tx = self.mesh.tangents[:, 0]
        ty = self.mesh.tangents[:, 1]
        phi = np.arctan2(ty, tx)
        
        # Ensure phi is in [0, 2pi] to match reference implementation logic if needed,
        # though atan2 returns [-pi, pi]. The reference implementation does:
        # if (phi[i] < 0): phi[i] = phi[i] + 2*np.pi
        phi = np.where(phi < 0, phi + 2*np.pi, phi)
        
        I = np.zeros((n_panels, n_panels))
        J = np.zeros((n_panels, n_panels))
        
        # Loop for now (optimization can come later)
        # i: target (control point)
        # j: source (panel)
        for i in range(n_panels):
            for j in range(n_panels):
                if i == j:
                    # Self-influence is handled separately in system assembly for I (usually pi),
                    # but pure geometric integral of 1/r over line for normal is usually 0 if flat?
                    # The reference code says: "if j!=i: ... else: I=0, J=0"
                    # And then adds pi to A[i,i].
                    # I[i,j] geometric integral is integral of (normal dot grad(ln r)).
                    # For a flat panel, the normal velocity induced by itself is 0 from the integral
                    # (since point is on the panel, normal is perpendicular to flow along panel),
                    # BUT the limit approach gives 2pi*sigma/2 = pi*sigma? 
                    # Reference code skips i==j in COMPUTE_IJ, returns 0.
                    continue
                
                # Intermediate terms from Katz & Plotkin / Reference
                # Relative coordinates
                dx_val = XC[i] - XB[j]
                dy_val = YC[i] - YB[j]
                
                # Transform to panel j local coordinates
                A = -dx_val * np.cos(phi[j]) - dy_val * np.sin(phi[j])
                B = dx_val**2 + dy_val**2
                
                # Orientation differences
                Cn = np.sin(phi[i] - phi[j])
                Dn = -dx_val * np.sin(phi[i]) + dy_val * np.cos(phi[i])
                Ct = -np.cos(phi[i] - phi[j])
                Dt = dx_val * np.cos(phi[i]) + dy_val * np.sin(phi[i])
                
                E_sq = B - A**2
                if E_sq <= 0: # Numerical stability
                    E = 0
                else:
                    E = np.sqrt(E_sq)
                
                if E == 0:
                    I[i, j] = 0
                    J[i, j] = 0
                    continue
                
                # Compute I (Normal)
                term1_I = 0.5 * Cn * np.log((S[j]**2 + 2*A*S[j] + B) / B)
                term2_I = ((Dn - A*Cn) / E) * (math.atan2((S[j] + A), E) - math.atan2(A, E))
                I[i, j] = term1_I + term2_I
                
                # Compute J (Tangential)
                term1_J = 0.5 * Ct * np.log((S[j]**2 + 2*A*S[j] + B) / B)
                term2_J = ((Dt - A*Ct) / E) * (math.atan2((S[j] + A), E) - math.atan2(A, E))
                J[i, j] = term1_J + term2_J
                
        return I, J

    def solve(self):
        """
        Solve the linear system for source strengths and compute flow properties.
        """
        n_panels = len(self.mesh.panels)
        I, J = self.compute_geometric_integrals()
        
        # Construct Linear System: A * sigma = b
        # A[i, j] = I[i, j] / (2*pi) ?
        # Reference SP_Circle.py: 
        #   A[i,j] = I[i,j] (for i!=j)
        #   A[i,i] = pi
        #   b[i] = -Vinf * 2 * pi * cos(beta_i)
        # This implies the equation is: sum(sigma_j * I_ij) + sigma_i * pi = -b_i
        # Let's stick to reference implementation exactly.
        
        A = np.copy(I)
        np.fill_diagonal(A, np.pi)
        
        # Compute beta (angle between panel normal and freestream)
        # normals angle: delta = phi + pi/2
        # beta = delta - aoa
        # Alternatively: beta is angle between n and V_inf.
        # cos(beta) = n . (V_inf / |V_inf|)
        # V_inf vector = (cos(aoa), sin(aoa))
        nx = self.mesh.normals[:, 0]
        ny = self.mesh.normals[:, 1]
        
        # Dot product
        # cos(beta) = nx*cos(aoa) + ny*sin(aoa)
        # Note: Reference uses b = -Vinf * 2 * pi * cos(beta)
        
        v_inf_vec = np.array([np.cos(self.aoa), np.sin(self.aoa)])
        cos_beta = nx * v_inf_vec[0] + ny * v_inf_vec[1]
        
        b = -self.v_inf * 2 * np.pi * cos_beta
        
        # Solve for source strengths
        try:
            self.sigma = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback for singular matrix (e.g. flat plate with no thickness might fail SPM?)
            # SPM requires closed body with thickness usually.
            print("Warning: Singular matrix in SourcePanelSolver. Check geometry.")
            self.sigma = np.zeros(n_panels)

        # Compute Tangential Velocities and Cp
        # Vt_i = Vinf * sin(beta_i) + sum(sigma_j / (2*pi) * J_ij)
        # Note: sin(beta) corresponds to tangential component of freestream?
        # t . V_inf. 
        # t is (nx, ny) rotated -90 deg? Or +90?
        # Mesh.tangents is usually CCW. n is outward.
        # If n = (cos d, sin d), t = (-sin d, cos d) usually?
        # Let's check mesh.py or just use the reference logic.
        # Ref: Vt[i] = Vinf * sin(beta[i]) + ...
        # beta = delta - aoa. sin(beta) = sin(delta - aoa).
        # delta is normal angle.
        # Tangent angle is usually delta - pi/2 (if n is +90 from t) or delta + pi/2.
        # If we trust the projection: V_inf_tangent = V_inf . t_i
        
        tx = self.mesh.tangents[:, 0]
        ty = self.mesh.tangents[:, 1]
        v_inf_tan = self.v_inf * (tx * v_inf_vec[0] + ty * v_inf_vec[1])
        
        # Summation term: sum( sigma[j] * J[i,j] ) / (2*pi)
        induced_tan = (self.sigma @ J.T) / (2 * np.pi) # Check matrix mult order. J[i,j] -> sum_j J[i,j]*sigma[j]
        # J shape (N,N), sigma shape (N,). (N,N) @ (N,) -> (N,) sum over last axis (j). Correct.
        # But verify: sum_j (sigma[j] * J[i,j]) -> dot product of row i of J with sigma.
        # standard np.dot(J, sigma) does this.
        
        induced_tan = np.dot(J, self.sigma) / (2 * np.pi)
        
        self.Vt = v_inf_tan + induced_tan
        self.Cp = 1.0 - (self.Vt / self.v_inf)**2
        
        # Store results in mesh
        self.mesh.cell_data['source_strength'] = self.sigma
        self.mesh.cell_data['Vt'] = self.Vt
        self.mesh.cell_data['Cp'] = self.Cp

