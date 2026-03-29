import numpy as np

def compute_analytical_HG(nodes_eval, nodes_b, normals_b, lengths_b):
    """
    nodes_eval: (M, 2)
    nodes_b: (N, 2) midpoints
    normals_b: (N, 2) outward normals
    lengths_b: (N,)
    """
    M = len(nodes_eval)
    N = len(nodes_b)
    
    # Tangents (N, 2)
    tx = -normals_b[:, 1]
    ty = normals_b[:, 0]
    
    # Endpoints
    half_L = lengths_b / 2
    p1x = nodes_b[:, 0] - half_L * tx
    p1y = nodes_b[:, 1] - half_L * ty
    p2x = nodes_b[:, 0] + half_L * tx
    p2y = nodes_b[:, 1] + half_L * ty
    
    # Relative vectors: shape (M, N)
    rx1 = nodes_eval[:, 0:1] - p1x[None, :]
    ry1 = nodes_eval[:, 1:2] - p1y[None, :]
    
    rx2 = nodes_eval[:, 0:1] - p2x[None, :]
    ry2 = nodes_eval[:, 1:2] - p2y[None, :]
    
    # Local coordinates of eval points relative to panel start (p1)
    # x_loc = rx1 * tx + ry1 * ty
    x_loc = rx1 * tx[None, :] + ry1 * ty[None, :]
    y_loc = rx1 * normals_b[:, 0][None, :] + ry1 * normals_b[:, 1][None, :]
    
    r1_sq = x_loc**2 + y_loc**2
    r2_sq = (x_loc - lengths_b[None, :])**2 + y_loc**2
    
    # theta1 and theta2
    theta1 = np.arctan2(y_loc, x_loc)
    theta2 = np.arctan2(y_loc, x_loc - lengths_b[None, :])
    
    dtheta = theta2 - theta1
    # Wrap dtheta to [-pi, pi]
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    
    H = -dtheta / (2 * np.pi)
    
    r1 = np.sqrt(np.maximum(r1_sq, 1e-24))
    r2 = np.sqrt(np.maximum(r2_sq, 1e-24))
    
    L = lengths_b[None, :]
    val = (x_loc * np.log(r1) - (x_loc - L) * np.log(r2) + y_loc * dtheta - L)
    G = -val / (2 * np.pi)
    
    # Handle self-influence (when nodes_eval == nodes_b)
    # If eval point is on the panel, y_loc=0. 
    # For H, if y_loc=0 and 0 < x_loc < L, dtheta = pi, H = -0.5
    # For BDIM, H_ii = 0.5 because the matrix equation is c(x)T + H T = G q
    # Usually we add c(x) = 0.5 to the diagonal. 
    # So if M==N, we just set the diagonal.
    if M == N:
        diag_idx = np.arange(N)
        H[diag_idx, diag_idx] = 0.5
        # G diagonal: (L / 2pi) * (1 - ln(L/2))
        G[diag_idx, diag_idx] = (lengths_b / (2 * np.pi)) * (1.0 - np.log(lengths_b / 2.0))
        
    return H, G

nodes_eval = np.array([[0, 5e-5]])
nodes_b = np.array([[0, 0]])
normals_b = np.array([[0, 1]])
lengths_b = np.array([0.012])

H, G = compute_analytical_HG(nodes_eval, nodes_b, normals_b, lengths_b)
print("H:", H[0, 0])
print("G:", G[0, 0])
