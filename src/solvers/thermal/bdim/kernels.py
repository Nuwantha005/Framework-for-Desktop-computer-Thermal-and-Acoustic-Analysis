import numpy as np
from numpy.typing import NDArray

def compute_r_and_vec(x: NDArray, y: NDArray) -> tuple[float, NDArray]:
    """Compute distance r and vector (y - x) from source x to field y."""
    vec = y - x
    r = np.linalg.norm(vec)
    return r, vec

def temp_fundamental(x: NDArray, y: NDArray) -> float:
    """T*(x, y) = (1 / 2\pi) * ln(1/r)"""
    r, _ = compute_r_and_vec(x, y)
    if r < 1e-12:
        return 0.0  # Singularity handled by diagonal approximations
    return (1.0 / (2.0 * np.pi)) * np.log(1.0 / r)

def temp_derivative(x: NDArray, y: NDArray) -> NDArray:
    """T^*_{,i}(x, y) = \partial T^* / \partial y_i = (-1 / 2\pi r^2) * (y_i - x_i)"""
    r, vec = compute_r_and_vec(x, y)
    if r < 1e-12:
        return np.zeros_like(vec)
    return (-1.0 / (2.0 * np.pi * r**2)) * vec

def temp_normal_derivative(x: NDArray, y: NDArray, n: NDArray) -> float:
    """T^*_{,n}(x, y) = T^*_{,i} * n_i"""
    grad = temp_derivative(x, y)
    return float(np.dot(grad, n))
