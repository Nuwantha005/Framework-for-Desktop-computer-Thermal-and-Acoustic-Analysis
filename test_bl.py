import numpy as np
from src.solvers.boundary_layer.base import BoundaryLayerSolver
from src.solvers.boundary_layer.profiles.thwaites import ThwaitesProfile

s = np.linspace(0, 1, 100)
Ue = s  # Ue = s, so dUe/ds = 1

solver = BoundaryLayerSolver(
    edge_velocity=Ue,
    arc_length=s,
    nu=1e-5,
    profile=ThwaitesProfile()
)

res = solver.solve()
print("s:", res.s[:10])
print("cf:", res.cf[:10])
print("Ue:", res.Ue[:10])
print("theta:", res.theta[:10])
