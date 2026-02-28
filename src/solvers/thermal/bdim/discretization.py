import numpy as np
from numpy.typing import NDArray

from .kernels import temp_fundamental, temp_derivative, temp_normal_derivative


def assemble_boundary_matrices(nodes_b: NDArray, normals_b: NDArray, lengths_b: NDArray):
    """
    Assemble characteristic BEM matrices [H] and [G] for the boundary equations.
    """
    N = len(nodes_b)
    H = np.zeros((N, N))
    G = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i == j:
                # Diagonal analytical treatment: c(x) = 0.5 for a smooth boundary
                H[i, j] = 0.5
                L = lengths_b[i]
                # Logarithmic integral analytically solved for the auto-influence panel
                G[i, j] = (L / (2.0 * np.pi)) * (1.0 - np.log(L / 2.0))
            else:
                x = nodes_b[i]
                y = nodes_b[j]
                n = normals_b[j]
                L = lengths_b[j]
                
                # Simple collocation approximation (constant property across panel)
                H[i, j] = temp_normal_derivative(x, y, n) * L
                G[i, j] = -temp_fundamental(x, y) * L

    return H, G


def assemble_domain_matrices(nodes_eval: NDArray, nodes_domain: NDArray, areas_domain: NDArray):
    """
    Assemble the volume mapping matrix [E] defining the internal thermal/dissipation effect.
    Returns array of shape (M, K, 2).
    """
    M = len(nodes_eval)
    K = len(nodes_domain)
    E = np.zeros((M, K, 2))

    for i in range(M):
        for j in range(K):
            if np.allclose(nodes_eval[i], nodes_domain[j], atol=1e-12):
                # Principal value integrals handling is required for rigorous treatment 
                # (For simple object scope: ignored/set to zero on internal intersection)
                E[i, j, :] = 0.0
            else:
                grad = temp_derivative(nodes_eval[i], nodes_domain[j])
                E[i, j, :] = -grad * areas_domain[j]
                
    return E


def assemble_boundary_domain_coupling(nodes_eval: NDArray, nodes_b: NDArray, normals_b: NDArray, lengths_b: NDArray):
    """
    Assemble explicit mapping defining domain source projections directly across boundary constraints.
    Returns (M, N, 2)
    """
    M = len(nodes_eval)
    N = len(nodes_b)
    EC_b = np.zeros((M, N, 2))
    
    for i in range(M):
        for j in range(N):
            x = nodes_eval[i]
            y = nodes_b[j]
            n = normals_b[j]
            L = lengths_b[j]
            
            # Diagonal self-influence
            if M == N and i == j:
                T_star = (1.0 / (2.0 * np.pi)) * np.log(2.0 / L) # Generic diagonal approx
            else:
                T_star = temp_fundamental(x, y)
                
            EC_b[i, j, 0] = T_star * n[0] * L
            EC_b[i, j, 1] = T_star * n[1] * L
            
    return EC_b
