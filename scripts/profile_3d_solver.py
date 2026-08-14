import cProfile
import pstats
import io
import time
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from core.io.case_loader import CaseLoader
from solvers.factory import SolverFactory

def profile_solver():
    print("Loading mesh level 1...")
    scene, config = CaseLoader.load("cases/sphere_flow/case.yaml", mesh_level_index=1)
    mesh = scene.assemble()
    
    freestream = config.get_freestream_velocity()
    solver = SolverFactory.create(
        config=config.solver,
        mesh=mesh,
        v_inf=freestream[0],
        aoa=0.0
    )
    solver._v_inf = np.asarray(freestream, dtype=np.float64)
    
    print(f"Starting solve on {mesh.num_panels} panels...")
    
    # Warmup Numba 
    print("Warming up JIT...")
    _ = solver._compute_influence_matrix()
    from src.solvers.panel3d.influences import compute_all_velocities_influence
    _ = compute_all_velocities_influence(solver._mesh.centers[:2], solver._mesh.nodes, solver._mesh.panels, np.ones(solver._mesh.num_panels))
    
    # Run with cProfile
    pr = cProfile.Profile()
    pr.enable()
    
    t0 = time.time()
    influence_matrix = solver._compute_influence_matrix()
    t1 = time.time()
    
    solver._influence_matrix = influence_matrix
    strengths = solver._solve_linear_system(influence_matrix)
    t2 = time.time()
    
    solver._surface_velocity = solver._compute_surface_velocity(strengths)
    solver._solved = True
    t3 = time.time()
    
    pr.disable()
    
    print("-" * 40)
    print("Manual Timing Breakdown:")
    print(f"1. Build Influence Matrix: {t1 - t0:.4f} sec")
    print(f"2. Solve Linear System:    {t2 - t1:.4f} sec")
    print(f"3. Compute Surfac Velocity:{t3 - t2:.4f} sec")
    print(f"Total Time:                {t3 - t0:.4f} sec")
    print("-" * 40)
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats(15)
    print("cProfile Breakdown (Top 15 operations):")
    print(s.getvalue())

if __name__ == '__main__':
    profile_solver()
