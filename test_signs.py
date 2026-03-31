import sys
sys.path.append("src")
from core.io.case_loader import CaseLoader
from postprocessing.surface import SurfaceDataExtractor
import numpy as np

case = CaseLoader.load_case("cases/rounded_square", mesh_level_index=4)
solver = case.create_solver()
solver.solve()
extractor = SurfaceDataExtractor(case.mesh, solver)
surface = extractor.extract(arc_length=False)

# Front face is x ~ -0.5
front_mask = np.isclose(surface.x, -0.5, atol=0.05)
y_front = surface.y[front_mask]
vt_front = surface.Vt[front_mask]

# Sort by y
sort_idx = np.argsort(y_front)
y_front = y_front[sort_idx]
vt_front = vt_front[sort_idx]

for y, vt in zip(y_front, vt_front):
    print(f"y: {y:6.3f} -> Vt: {vt:8.3f}")
