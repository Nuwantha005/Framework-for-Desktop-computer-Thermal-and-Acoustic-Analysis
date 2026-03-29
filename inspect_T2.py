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
bl_field = list(path_result.fields.values())[0]
bdim_input = extract_bdim_input_from_bl_field(path_result, bl_field)
config = BDIMConfig(T_inf=300.0, q_wall=500.0)

# Modify solver logic directly for test
bdim_solver = BDIMThermalSolver(bdim_input, config)
N = len(bdim_input.nodes_b)
K = len(bdim_input.nodes_domain)
M, Ny = bdim_input.grid_shape

from solvers.thermal.bdim.discretization import assemble_boundary_matrices, assemble_domain_matrices, assemble_boundary_domain_coupling

H, G = assemble_boundary_matrices(bdim_input.nodes_b, bdim_input.normals_b, bdim_input.lengths_b)
E_b_dom = assemble_domain_matrices(bdim_input.nodes_b, bdim_input.nodes_domain, bdim_input.areas_domain)
EC_b = assemble_boundary_domain_coupling(bdim_input.nodes_b, bdim_input.nodes_b, bdim_input.normals_b, bdim_input.lengths_b)
E_I_dom = assemble_domain_matrices(bdim_input.nodes_domain, bdim_input.nodes_domain, bdim_input.areas_domain)
EC_I = assemble_boundary_domain_coupling(bdim_input.nodes_domain, bdim_input.nodes_b, bdim_input.normals_b, bdim_input.lengths_b)

dx_local = np.sqrt(bdim_input.areas_domain)
u_mag = np.linalg.norm(bdim_input.u_domain, axis=1)
k_eff_I = np.maximum(bdim_solver.k, bdim_solver.rho * bdim_solver.cp * u_mag * dx_local / 0.5)

u_mag_b = np.linalg.norm(bdim_input.u_b, axis=1)
u_mag_b_eff = np.maximum(u_mag_b, 1.0)
dx_b = bdim_input.lengths_b
k_eff_b = np.maximum(bdim_solver.k, bdim_solver.rho * bdim_solver.cp * u_mag_b_eff * dx_b / 0.5)

c_w_I = np.zeros((K, 2))
c_w_b = np.zeros((N, 2))

rho_cv_u_b = (bdim_solver.rho * bdim_solver.cp / k_eff_b[:, None]) * bdim_input.u_b
rho_cv_u_I = (bdim_solver.rho * bdim_solver.cp / k_eff_I[:, None]) * bdim_input.u_domain

D_b = np.sum(E_b_dom * rho_cv_u_I[None, :, :], axis=2)
D_I = np.sum(E_I_dom * rho_cv_u_I[None, :, :], axis=2)
np.fill_diagonal(D_I, 0.0)

B_b = np.sum(EC_b * rho_cv_u_b[None, :, :], axis=2)
B_I = np.sum(EC_I * rho_cv_u_b[None, :, :], axis=2)

const_b = np.sum(EC_b * c_w_b[:, None, :], axis=2).sum(axis=1) - np.sum(E_b_dom * c_w_I[None, :, :], axis=2).sum(axis=1)
const_I = np.sum(EC_I * c_w_b[None, :, :], axis=2).sum(axis=1) - np.sum(E_I_dom * c_w_I[None, :, :], axis=2).sum(axis=1)

from solvers.thermal.bdim.discretization import _compute_distances, _temp_fundamental_vectorized, _temp_derivative_vectorized
r_I, vec_I = _compute_distances(bdim_input.nodes_domain, bdim_input.nodes_b)
T_star_I = _temp_fundamental_vectorized(r_I)
grad_T_star_I = _temp_derivative_vectorized(r_I, vec_I)

G_I = T_star_I * bdim_input.lengths_b[None, :]
H_I = np.sum(grad_T_star_I * bdim_input.normals_b[None, :, :], axis=2)
H_I = H_I * bdim_input.lengths_b[None, :]

q_b = np.full(N, config.q_wall)
q_b_k = q_b / k_eff_b
Y_b = G @ q_b_k + const_b
Y_I = G_I @ q_b_k + const_I

Sys_mat = np.block([
    [H + B_b, D_b],
    [H_I + B_I, np.eye(K) + D_I]
])

Sys_rhs = np.concatenate([Y_b, Y_I])

# Enforce BCs
outer_idx = np.arange(M) * Ny + (Ny - 1)
inflow_idx = np.arange(1, Ny) # ONLY off-wall inflow! Skip j=0.
bc_idx = np.unique(np.concatenate([outer_idx, inflow_idx]))

sys_bc_idx = N + bc_idx
Sys_mat[sys_bc_idx, :] = 0.0
Sys_mat[sys_bc_idx, sys_bc_idx] = 1.0
Sys_rhs[sys_bc_idx] = 0.0

# ENFORCE wall compatibility: T_I(j=0) = T_b(i)
wall_domain_idx = np.arange(M) * Ny
sys_wall_domain_idx = N + wall_domain_idx

Sys_mat[sys_wall_domain_idx, :] = 0.0
Sys_mat[sys_wall_domain_idx, sys_wall_domain_idx] = 1.0
Sys_mat[sys_wall_domain_idx, np.arange(M)] = -1.0
Sys_rhs[sys_wall_domain_idx] = 0.0

solution = np.linalg.solve(Sys_mat, Sys_rhs)
T_b = solution[:N] + config.T_inf
T_I = solution[N:] + config.T_inf

T_field = T_I.reshape(M, Ny)

print('min T:', np.min(T_field))
print('max T:', np.max(T_field))

min_idx = np.unravel_index(np.argmin(T_field, axis=None), T_field.shape)
print(f'min T is at i={min_idx[0]}, j={min_idx[1]} (s={bdim_input.arc_length[min_idx[0]]:.4f})')
for di in [-1, 0, 1]:
    for dj in [-1, 0, 1]:
        i, j = min_idx[0]+di, min_idx[1]+dj
        if 0 <= i < T_field.shape[0] and 0 <= j < T_field.shape[1]:
            print(f'T[{i}, {j}] = {T_field[i, j]:.2f}')

# Check the row for i=1, j=1 in Sys_mat
row_idx = N + 1 * Ny + 1
row = Sys_mat[row_idx, :]
print("\nRow for i=1, j=1 in Sys_mat:")
print("Diagonal element:", row[row_idx])
print("Max abs element:", np.max(np.abs(row)))
print("Min element:", np.min(row))
print("Sum of off-diagonal abs elements in domain:", np.sum(np.abs(row[N:])) - abs(row[row_idx]))

# which elements are the largest?
largest_idx = np.argsort(np.abs(row))[-10:]
for idx in largest_idx:
    if idx < N:
        print(f"  Boundary node {idx}: value {row[idx]}")
    else:
        dom_idx = idx - N
        i_dom = dom_idx // Ny
        j_dom = dom_idx % Ny
        print(f"  Domain node i={i_dom}, j={j_dom}: value {row[idx]}")

print(f"H_I[row, 1] = {H_I[row_idx-N, 1]}")
print(f"B_I[row, 1] = {B_I[row_idx-N, 1]}")
print(f"Max E_b_dom: {np.max(np.abs(E_b_dom))}")
print(f"Max E_I_dom: {np.max(np.abs(E_I_dom))}")
