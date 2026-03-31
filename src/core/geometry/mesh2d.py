"""
2D mesh data structure for panel methods.

Defines line-segment panels for 2D flows (airfoils, cylinders, etc.).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
from numpy.typing import NDArray

from .mesh_base import MeshBase


@dataclass(repr=False)
class Mesh2D(MeshBase):
    """
    2D panel mesh with line segment panels.
    
    Extends MeshBase for 2D panel methods. Panels are line segments
    defined by pairs of node indices. All coordinates use (N, 3) arrays
    with z=0 for consistency with 3D extension.
    
    Attributes:
        nodes: Node coordinates (N, 3) with z=0
        panels: Panel connectivity (P, 2) - pairs of node indices
        component_ids: Component ID for each panel (P,)
        centers: Panel center points (P, 3) - computed
        normals: Panel outward unit normals (P, 3) - computed
        tangents: Panel tangent vectors (P, 3) - computed (2D specific)
        areas: Panel lengths (P,) - computed
        cell_data: Results storage dict
    
    Note:
        For backward compatibility, this class is also exported as `Mesh`
        from the geometry module.
    """
    
    # 2D-specific computed geometry
    tangents: Optional[NDArray[np.float64]] = None      # (P, 3)
    
    @property
    def dimension(self) -> int:
        """Return mesh dimension (2)."""
        return 2
    
    def _validate(self) -> None:
        """Validate 2D mesh properties."""
        super()._validate()
        
        expected_panel_size = 2
        if self.panels.shape[1] != expected_panel_size:
            raise ValueError(
                f"For 2D mesh, panels must have shape (P, 2), got {self.panels.shape}"
            )
    
    def compute_geometry(self) -> None:
        """Compute panel centers, normals, tangents, and lengths."""
        num_panels = self.num_panels
        
        # Initialize arrays
        self.centers = np.zeros((num_panels, 3), dtype=np.float64)
        self.normals = np.zeros((num_panels, 3), dtype=np.float64)
        self.tangents = np.zeros((num_panels, 3), dtype=np.float64)
        self.areas = np.zeros(num_panels, dtype=np.float64)
        
        for i in range(num_panels):
            # Get panel endpoints
            n1_idx, n2_idx = self.panels[i]
            p1 = self.nodes[n1_idx]
            p2 = self.nodes[n2_idx]
            
            # Center
            self.centers[i] = 0.5 * (p1 + p2)
            
            # Tangent (p1 → p2)
            tangent = p2 - p1
            length = np.linalg.norm(tangent)
            
            if length < 1e-14:
                raise ValueError(f"Panel {i} has zero length")
            
            self.tangents[i] = tangent / length
            self.areas[i] = length  # "area" is length for 2D
            
            # Normal (rotate tangent 90° CCW in xy-plane)
            # For 2D in xy-plane: n = (t_y, -t_x, 0)
            t = self.tangents[i]
            self.normals[i] = np.array([t[1], -t[0], 0.0])


# Backward compatibility alias
Mesh = Mesh2D
