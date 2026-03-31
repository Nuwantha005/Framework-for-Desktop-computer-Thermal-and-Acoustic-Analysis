import sys
sys.path.append("src")
from core.io.case_loader import CaseLoader
from solvers.boundary_layer.runner import BoundaryLayerRunner
import numpy as np

case = CaseLoader.load_case("cases/cylinder_flow", mesh_level_index=3)
solver = case.create_solver()
solver.solve()
runner = BoundaryLayerRunner(case, solver)
bl = runner.run(["pohlhausen"])

res = bl.upper.results["Pohlhausen"]
valid = ~np.isnan(res.theta)
print(f"Total panels: {len(res.s)}")
print(f"Valid panels before separation: {np.sum(valid)}")
print(f"Separation at s = {res.s[valid][-1]:.4f} (max s = {res.s[-1]:.4f})")
