"""
3D mesh data structure for panel methods.

Defines quadrilateral surface panels for 3D flows (spheres, bodies, etc.).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
from numpy.typing import NDArray

from .mesh_base import MeshBase


@dataclass(repr=False)
class Mesh3D(MeshBase):
    """
    3D surface mesh with quadrilateral panels.
    
    Extends MeshBase for 3D panel methods. Panels are quadrilaterals
    defined by four node indices in CCW order (when viewed from outside).
    
    Attributes:
        nodes: Node coordinates (N, 3)
        panels: Panel connectivity (P, 4) - four node indices per panel
        component_ids: Component ID for each panel (P,)
        centers: Panel center points (P, 3) - computed
        normals: Panel outward unit normals (P, 3) - computed
        tangent1: First tangent direction (P, 3) - computed (along first edge)
        tangent2: Second tangent direction (P, 3) - computed (perpendicular to tangent1)
        areas: Panel areas (P,) - computed
        cell_data: Results storage dict
    """
    
    # 3D-specific computed geometry
    tangent1: Optional[NDArray[np.float64]] = None      # (P, 3)
    tangent2: Optional[NDArray[np.float64]] = None      # (P, 3)
    
    @property
    def dimension(self) -> int:
        """Return mesh dimension (3)."""
        return 3
    
    def _validate(self) -> None:
        """Validate 3D mesh properties."""
        super()._validate()
        
        expected_panel_size = 4
        if self.panels.shape[1] != expected_panel_size:
            raise ValueError(
                f"For 3D mesh, panels must have shape (P, 4), got {self.panels.shape}"
            )
    
    def compute_geometry(self) -> None:
        """Compute panel centers, normals, tangents, and areas."""
        num_panels = self.num_panels
        
        # Initialize arrays
        self.centers = np.zeros((num_panels, 3), dtype=np.float64)
        self.normals = np.zeros((num_panels, 3), dtype=np.float64)
        self.tangent1 = np.zeros((num_panels, 3), dtype=np.float64)
        self.tangent2 = np.zeros((num_panels, 3), dtype=np.float64)
        self.areas = np.zeros(num_panels, dtype=np.float64)
        
        for i in range(num_panels):
            # Get panel corner nodes (CCW ordering when viewed from outside)
            n1_idx, n2_idx, n3_idx, n4_idx = self.panels[i]
            p1 = self.nodes[n1_idx]
            p2 = self.nodes[n2_idx]
            p3 = self.nodes[n3_idx]
            p4 = self.nodes[n4_idx]
            
            # Center (average of corners)
            self.centers[i] = 0.25 * (p1 + p2 + p3 + p4)
            
            # Normal via cross product of diagonals
            # For planar quad: n = (p3 - p1) × (p4 - p2)
            d1 = p3 - p1
            d2 = p4 - p2
            normal = np.cross(d1, d2)
            normal_mag = np.linalg.norm(normal)
            
            if normal_mag < 1e-14:
                raise ValueError(f"Panel {i} has degenerate normal (colinear diagonals)")
            
            self.normals[i] = normal / normal_mag
            
            # Area (shoelace formula approximation for quad)
            # Using split triangle method: area = 0.5 * |d1 × d2|
            self.areas[i] = 0.5 * normal_mag
            
            # First tangent (first edge direction)
            edge1 = p2 - p1
            edge1_mag = np.linalg.norm(edge1)
            if edge1_mag > 1e-14:
                self.tangent1[i] = edge1 / edge1_mag
            else:
                # Fallback: use second edge
                edge2 = p3 - p2
                edge2_mag = np.linalg.norm(edge2)
                if edge2_mag > 1e-14:
                    self.tangent1[i] = edge2 / edge2_mag
                else:
                    raise ValueError(f"Panel {i} has degenerate edges")
            
            # Second tangent (perpendicular to normal and tangent1)
            self.tangent2[i] = np.cross(self.normals[i], self.tangent1[i])
    
    def to_pyvista(self):
        """
        Convert mesh to PyVista PolyData for visualization.
        
        Returns:
            pyvista.PolyData object with quads
        
        Raises:
            ImportError: If pyvista is not installed
        """
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("PyVista is required for VTK export: pip install pyvista")
        
        # Build faces array: [4, i0, i1, i2, i3, 4, i0, i1, ...]
        faces = np.hstack([
            np.column_stack([
                np.full(self.num_panels, 4, dtype=np.int32),
                self.panels
            ]).ravel()
        ])
        
        polydata = pv.PolyData(self.nodes, faces)
        
        # Add cell data
        for name, data in self.cell_data.items():
            polydata.cell_data[name] = data
        
        return polydata
    
    def save_vtk(self, path: str) -> None:
        """
        Save mesh to VTK file for ParaView visualization.
        
        Args:
            path: Output file path (.vtu, .vtk, or .vtp)
        """
        polydata = self.to_pyvista()
        polydata.save(path)
