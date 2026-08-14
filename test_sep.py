import sys
sys.path.append("src")
from core.io.case_loader import CaseLoader
from solvers.boundary_layer.runner import BoundaryLayerRunner
import numpy as np

case = CaseLoader.load_case("cases/rounded_square", mesh_level_index=4)
solver = case.create_solver()
solver.solve()
runner = BoundaryLayerRunner(case, solver)
bl = runner.run(["pohlhausen"])

res = bl.upper.results["Pohlhausen"]
s_sep = res.s[-1]
idx_sep = len(res.s) - 1
panel_idx = bl.upper.panel_indices[idx_sep]
x_sep = case.mesh.centers[panel_idx, 0]
y_sep = case.mesh.centers[panel_idx, 1]

print(f"Separation at s = {s_sep:.4f}")
print(f"Separation coordinates: x = {x_sep:.4f}, y = {y_sep:.4f}")

# Let's print Ue and dUe/ds near separation
for i in range(max(0, idx_sep - 5), idx_sep + 1):
    print(f"s: {res.s[i]:.4f}, Ue: {res.Ue[i]:.4f}, theta: {res.theta[i]:.6f}, H: {res.H[i]:.4f}")

print("\nLet's find the actual separation point (where theta becomes nan):")
valid_indices = np.where(~np.isnan(res.theta))[0]
last_valid = valid_indices[-1]
panel_idx = bl.upper.panel_indices[last_valid]
print(f"Last valid s = {res.s[last_valid]:.4f}")
print(f"Last valid coordinates: x = {case.mesh.centers[panel_idx, 0]:.4f}, y = {case.mesh.centers[panel_idx, 1]:.4f}")
print(f"First invalid s = {res.s[last_valid+1]:.4f} (Ue = {res.Ue[last_valid+1]:.4f})")
