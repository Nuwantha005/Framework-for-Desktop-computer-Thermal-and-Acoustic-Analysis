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

Vt = surface.Vt
print(f"Min Vt: {np.min(Vt)}, Max Vt: {np.max(Vt)}")

signs = np.sign(Vt)
sign_changes = np.where(np.diff(np.concatenate([signs, [signs[0]]])) != 0)[0]
print(f"Sign changes: {sign_changes}")
for i in sign_changes:
    i_next = (i + 1) % len(Vt)
    print(f"Sign change at {i} -> {i_next}: Vt {Vt[i]:.4f} -> {Vt[i_next]:.4f}")
    
    x = case.mesh.nodes[i, 0]
    y = case.mesh.nodes[i, 1]
    print(f"  Approx pos: ({x:.4f}, {y:.4f})")

