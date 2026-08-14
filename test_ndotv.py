import sys
sys.path.append("src")
from core.io.case_loader import CaseLoader
import numpy as np

case = CaseLoader.load_case("cases/rounded_square", mesh_level_index=4)
solver = case.create_solver()

normals_2d = case.mesh.normals[:, :2]
v_inf_2d = solver.v_inf_vector[:2]
v_inf_unit = v_inf_2d / np.linalg.norm(v_inf_2d)

n_dot_v = normals_2d @ v_inf_unit

min_val = np.min(n_dot_v)
indices = np.where(np.isclose(n_dot_v, min_val))[0]
print(f"Minimum n.V is {min_val}")
print(f"Number of panels with this minimum: {len(indices)}")
print(f"First panel with this minimum: {indices[0]} at {case.mesh.centers[indices[0], :2]}")
print(f"Last panel with this minimum: {indices[-1]} at {case.mesh.centers[indices[-1], :2]}")
print(f"np.argmin returns: {np.argmin(n_dot_v)}")
