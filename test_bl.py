import sys
sys.path.append("src")
from core.io.case_loader import CaseLoader
from postprocessing.surface import SurfaceDataExtractor
from solvers.boundary_layer.runner import BoundaryLayerRunner

case = CaseLoader.load_case("cases/rounded_square", mesh_level_index=4)
solver = case.create_solver()
solver.solve()

runner = BoundaryLayerRunner(case, solver)
bl = runner.run()

print("Upper profiles:", bl.upper.results.keys())
print("Lower profiles:", bl.lower.results.keys())

