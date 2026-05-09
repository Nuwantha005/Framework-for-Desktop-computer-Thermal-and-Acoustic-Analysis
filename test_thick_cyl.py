import numpy as np
from core.geometry.mesh3d import Mesh3D

def generate_thick_cylinder(radius_in=0.06, radius_out=0.065, length=1.0, n_theta=8, n_length=2):
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    z = np.linspace(-length/2, length/2, n_length+1)
    
    nodes = []
    # Generate all inner nodes
    for zi in z:
        for t in theta:
            nodes.append([radius_in * np.cos(t), radius_in * np.sin(t), zi])
    inner_offset = 0
    
    # Generate all outer nodes
    for zi in z:
        for t in theta:
            nodes.append([radius_out * np.cos(t), radius_out * np.sin(t), zi])
    outer_offset = len(z) * n_theta
    
    nodes = np.array(nodes, dtype=np.float64)
    panels = []
    
    # Inner wall panels (normals should point inward, towards r=0)
    # If viewed from fluid (r=0), we want CCW ordering.
    # z goes up. t goes CCW.
    # From origin looking at wall, +t is right, +z is up.
    # So CCW is (t, z), (t+1, z), (t+1, z+1), (t, z+1) -> Wait.
    for i in range(n_length):
        for j in range(n_theta):
            n00 = inner_offset + i * n_theta + j
            n10 = inner_offset + i * n_theta + (j + 1) % n_theta
            n01 = inner_offset + (i + 1) * n_theta + j
            n11 = inner_offset + (i + 1) * n_theta + (j + 1) % n_theta
            
            # To face INWARD, viewing from origin, CCW: n00 -> n10 -> n11 -> n01
            panels.append([n00, n10, n11, n01])
            
    # Outer wall panels (normals should point outward)
    # Viewing from outside, +t is left, +z is up.
    for i in range(n_length):
        for j in range(n_theta):
            n00 = outer_offset + i * n_theta + j
            n10 = outer_offset + i * n_theta + (j + 1) % n_theta
            n01 = outer_offset + (i + 1) * n_theta + j
            n11 = outer_offset + (i + 1) * n_theta + (j + 1) % n_theta
            
            # To face OUTWARD, viewing from outside, CCW: n00 -> n01 -> n11 -> n10
            panels.append([n00, n01, n11, n10])
            
    # Bottom lip (z = -length/2) (normals point -z)
    # Viewing from -z (looking up), +t is CCW.
    # Inner is n00 (inner_offset + j), Outer is n00_out (outer_offset + j)
    for j in range(n_theta):
        in_0 = inner_offset + j
        in_1 = inner_offset + (j + 1) % n_theta
        out_0 = outer_offset + j
        out_1 = outer_offset + (j + 1) % n_theta
        
        # To face -z, viewing from bottom, CCW: in_0 -> out_0 -> out_1 -> in_1
        panels.append([in_0, out_0, out_1, in_1])
        
    # Top lip (z = length/2) (normals point +z)
    # Viewing from +z (looking down), +t is CCW.
    inner_top = inner_offset + n_length * n_theta
    outer_top = outer_offset + n_length * n_theta
    for j in range(n_theta):
        in_0 = inner_top + j
        in_1 = inner_top + (j + 1) % n_theta
        out_0 = outer_top + j
        out_1 = outer_top + (j + 1) % n_theta
        
        # To face +z, viewing from top, CCW: in_0 -> in_1 -> out_1 -> out_0
        panels.append([in_0, in_1, out_1, out_0])
        
    panels = np.array(panels, dtype=np.int32)
    mesh = Mesh3D(nodes, panels, np.zeros(len(panels), dtype=np.int32))
    return mesh

mesh = generate_thick_cylinder()
print(f"Num nodes: {len(mesh.nodes)}")
print(f"Num panels: {len(mesh.panels)}")

# Test normals
inner_normals = mesh.normals[:16]
outer_normals = mesh.normals[16:32]
bottom_normals = mesh.normals[32:40]
top_normals = mesh.normals[40:48]

print("Inner normals mean r:", np.mean(inner_normals[:, :2] * mesh.centers[:16, :2], axis=1))
print("Outer normals mean r:", np.mean(outer_normals[:, :2] * mesh.centers[16:32, :2], axis=1))
print("Bottom normals:", np.mean(bottom_normals, axis=0))
print("Top normals:", np.mean(top_normals, axis=0))

