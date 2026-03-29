import sys
from pathlib import Path
sys.path.insert(0, str(Path('/run/media/nuwa/Work/FYP/Code/panel-method-solver/src')))

from core.io.case_loader import CaseLoader
from solvers.boundary_layer.runner import BoundaryLayerRunner
from solvers.thermal.bdim.extraction import extract_bdim_input_from_bl_field
from solvers.thermal.bdim.solver import BDIMConfig, BDIMThermalSolver
import numpy as np

case = CaseLoader.load_case(Path('cases/cylinder_flow'), mesh_level_index=-1)
solver = case.create_solver()
solver.solve()

runner = BoundaryLayerRunner(case, solver)
bl = runner.run(profiles=['thwaites'], reconstruct=True)

path_result = bl.upper
print("Available fields:", path_result.fields.keys())
bl_field = list(path_result.fields.values())[0]

bdim_input = extract_bdim_input_from_bl_field(path_result, bl_field)

config = BDIMConfig(T_inf=300.0, q_wall=500.0)
bdim_solver = BDIMThermalSolver(bdim_input, config)
result = bdim_solver.solve()

field = result.field
T_field = field.T

print('min T:', np.min(T_field))
print('max T:', np.max(T_field))

min_idx = np.unravel_index(np.argmin(T_field, axis=None), T_field.shape)
print(f'min T is at i={min_idx[0]}, j={min_idx[1]} (s={field.s[min_idx[0]]:.4f}, y={field.y_normal[min_idx[0], min_idx[1]]:.4e})')

print('Surrounding T values:')
for di in [-2, -1, 0, 1, 2]:
    for dj in [-2, -1, 0, 1, 2]:
        i, j = min_idx[0]+di, min_idx[1]+dj
        if 0 <= i < T_field.shape[0] and 0 <= j < T_field.shape[1]:
            print(f'T[{i}, {j}] (s={field.s[i]:.4f}, y={field.y_normal[i, j]:.4e}) = {T_field[i, j]:.2f}')
