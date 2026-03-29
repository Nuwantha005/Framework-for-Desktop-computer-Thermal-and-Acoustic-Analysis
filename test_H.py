import numpy as np

def H_exact(x, y, x1, y1, x2, y2, nx, ny):
    dx = x2 - x1
    dy = y2 - y1
    L = np.sqrt(dx**2 + dy**2)
    tx, ty = dx/L, dy/L
    
    rx = x - x1
    ry = y - y1
    x_loc = rx * tx + ry * ty
    y_loc = rx * nx + ry * ny
    
    theta1 = np.arctan2(y_loc, x_loc)
    theta2 = np.arctan2(y_loc, x_loc - L)
    dtheta = theta2 - theta1
    
    # Handle branch cut if dtheta jumps by 2pi
    # We want the interior angle subtended by the panel
    while dtheta > np.pi: dtheta -= 2*np.pi
    while dtheta < -np.pi: dtheta += 2*np.pi
    
    return -dtheta / (2 * np.pi)

def G_exact(x, y, x1, y1, x2, y2, nx, ny):
    # G = int -1/(2pi) ln(r) ds
    dx = x2 - x1
    dy = y2 - y1
    L = np.sqrt(dx**2 + dy**2)
    tx, ty = dx/L, dy/L
    
    rx = x - x1
    ry = y - y1
    x_loc = rx * tx + ry * ty
    y_loc = rx * nx + ry * ny
    
    r1 = np.sqrt(x_loc**2 + y_loc**2)
    r2 = np.sqrt((x_loc - L)**2 + y_loc**2)
    
    # K&P Eq 10.22 (for unit source sigma)
    # phi_source = sigma/(4pi) * [x_loc ln(r1^2) - (x_loc-L) ln(r2^2) + 2 z_loc (theta2 - theta1) - 2L]
    # G = phi_source with sigma = -2 (because T* = -1/2pi ln r, while source is 1/2pi ln r)
    # Wait, source potential is 1/2pi int ln r ds.
    # So G = - int 1/2pi ln r ds = - phi_source(sigma=1)
    
    # Let's write the exact analytical integral of ln(r):
    theta1 = np.arctan2(y_loc, x_loc)
    theta2 = np.arctan2(y_loc, x_loc - L)
    dtheta = theta2 - theta1
    while dtheta > np.pi: dtheta -= 2*np.pi
    while dtheta < -np.pi: dtheta += 2*np.pi
    
    val = (x_loc * np.log(r1) - (x_loc - L) * np.log(r2) + y_loc * dtheta - L)
    return -val / (2 * np.pi)

def G_midpoint(x, y, x0, y0, L):
    rx = x - x0
    ry = y - y0
    r = np.sqrt(rx**2 + ry**2)
    return -np.log(r) * L / (2 * np.pi)

L = 0.012
y_eval = 5e-5
print(f"Exact H:    {H_exact(0, y_eval, -L/2, 0, L/2, 0, 0, 1)}")
print(f"Exact G:    {G_exact(0, y_eval, -L/2, 0, L/2, 0, 0, 1)}")
print(f"Midpoint G: {G_midpoint(0, y_eval, 0, 0, L)}")
