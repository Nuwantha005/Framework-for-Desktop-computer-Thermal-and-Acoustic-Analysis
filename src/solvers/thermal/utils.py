import numpy as np
from numpy.typing import NDArray

def compute_total_heat_rate(q: NDArray, s: NDArray) -> float:
    """
    Integrate local heat flux over the surface arc length to obtain
    the total heat rate (W/m spanwise) using the trapezoidal rule.
    """
    return float(np.trapz(q, s))
