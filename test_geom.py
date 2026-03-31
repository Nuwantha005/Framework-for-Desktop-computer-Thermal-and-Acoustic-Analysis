import sys
sys.path.append("src")
from core.io.case_loader import CaseLoader
case = CaseLoader.load_case("cases/rounded_square", mesh_level_index=4)
for i in range(125, 135):
    print(f"Panel {i}: {case.mesh.centers[i, 0]:.4f}, {case.mesh.centers[i, 1]:.4f}")
import numpy as np
from postprocessing.surface import SurfaceDataExtractor
from solvers.boundary_layer.runner import BoundaryLayerRunner

solver = case.create_solver()
solver.solve()
runner = BoundaryLayerRunner(case, solver)
bl = runner.run()

for side in ["upper", "lower"]:
    path = bl.sides[side]
    print(f"\n{side} path:")
    print(f"s array min: {path.s.min():.4f}, max: {path.s.max():.4f}")
    neg_count = np.sum(path.s < 0)
    print(f"Number of panels with s < 0: {neg_count} out of {len(path.s)}")
    
    # Let's print the (x, y) of the first panel where s >= 0
    idx_first_pos = np.where(path.s >= 0)[0][0]
    panel_idx = path.panel_indices[idx_first_pos]
    print(f"First panel with s >= 0: Index {panel_idx}, x={path.x[idx_first_pos]:.4f}, y={path.y[idx_first_pos]:.4f}")

