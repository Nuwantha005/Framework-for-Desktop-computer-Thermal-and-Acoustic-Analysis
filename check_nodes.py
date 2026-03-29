import sys
from pathlib import Path
sys.path.insert(0, str(Path('/run/media/nuwa/Work/FYP/Code/panel-method-solver/src')))

from core.io.case_loader import CaseLoader
from solvers.boundary_layer.runner import BoundaryLayerRunner
from solvers.thermal.bdim.extraction import extract_bdim_input_from_bl_field
import numpy as np

case = CaseLoader.load_case(Path('cases/cylinder_flow'), mesh_level_index=-1)
solver = case.create_solver()
solver.solve()

runner = BoundaryLayerRunner(case, solver)
bl = runner.run(profiles=['thwaites'], reconstruct=True)
path_result = bl.upper
bl_field = list(path_result.fields.values())[0]

bdim_input = extract_bdim_input_from_bl_field(path_result, bl_field)

nodes_b = bdim_input.nodes_b
nodes_domain = bdim_input.nodes_domain

M, Ny = bdim_input.grid_shape
for i in range(5):
    print(f'i={i}: nodes_b={nodes_b[i]}')
    print(f'      nodes_domain(j=0)={nodes_domain[i*Ny]}')
    print(f'      distance={np.linalg.norm(nodes_b[i] - nodes_domain[i*Ny])}')
