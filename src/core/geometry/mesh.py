"""
Mesh data structure for 2D and 3D panel methods.

Unified design: always use 3D arrays (z=0 for 2D) for seamless dimension transition.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
from numpy.typing import NDArray


@dataclass
class Mesh:
    """
    Unified mesh for 2D and 3D panel methods.
    
    Attributes:
        nodes: Node coordinates (N, 3) - ALWAYS 3D
        panels: Panel connectivity (P, k) where k=2 for 2D lines, k=4 for 3D quads
        dimension: Problem dimension (2 or 3)
        component_ids: Component ID for each panel (P,)
        centers: Panel center points (P, 3) - computed
        normals: Panel outward unit normals (P, 3) - computed
        tangents: Panel tangent vectors (P, 3) - computed
        areas: Panel lengths (2D) or areas (3D) (P,) - computed
        cell_data: Results storage dict, e.g., {'source_strength': array, 'Cp': array}
    """
    
    nodes: NDArray[np.float64]                    # (N, 3)
    panels: NDArray[np.int32]                     # (P, 2) or (P, 4)
    dimension: int                                # 2 or 3
    component_ids: NDArray[np.int32]              # (P,)
    
    # Computed geometry (set by compute_geometry())
    centers: Optional[NDArray[np.float64]] = None       # (P, 3)
    normals: Optional[NDArray[np.float64]] = None       # (P, 3)
    tangents: Optional[NDArray[np.float64]] = None      # (P, 3)
    areas: Optional[NDArray[np.float64]] = None         # (P,)
    
    # Results storage
    cell_data: Dict[str, NDArray] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate mesh data and compute geometry."""
        self._validate()
        self.compute_geometry()
    
    def _validate(self):
        """Check data consistency."""
        if self.dimension not in (2, 3):
            raise ValueError(f"dimension must be 2 or 3, got {self.dimension}")
        
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError(f"nodes must have shape (N, 3), got {self.nodes.shape}")
        
        if self.panels.ndim != 2:
            raise ValueError(f"panels must be 2D array, got {self.panels.ndim}D")
        
        expected_panel_size = 2 if self.dimension == 2 else 4
        if self.panels.shape[1] != expected_panel_size:
            raise ValueError(
                f"For dimension={self.dimension}, panels must have shape "
                f"(P, {expected_panel_size}), got {self.panels.shape}"
            )
        
        if self.component_ids.shape[0] != self.panels.shape[0]:
            raise ValueError(
                f"component_ids length ({self.component_ids.shape[0]}) must match "
                f"number of panels ({self.panels.shape[0]})"
            )
        
        # Check node indices are valid
        max_idx = np.max(self.panels)
        if max_idx >= self.nodes.shape[0]:
            raise ValueError(
                f"Panel references node index {max_idx} but only "
                f"{self.nodes.shape[0]} nodes exist"
            )
    
    @property
    def is_2d(self) -> bool:
        """Check if this is a 2D mesh."""
        return self.dimension == 2
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return self.nodes.shape[0]
    
    @property
    def num_panels(self) -> int:
        """Number of panels."""
        return self.panels.shape[0]
    
    def compute_geometry(self):
        """Compute panel centers, normals, tangents, and areas."""
        if self.dimension == 2:
            self._compute_geometry_2d()
        else:
            self._compute_geometry_3d()
    
    def _compute_geometry_2d(self):
        """Compute geometry for 2D line panels."""
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
    
    def _compute_geometry_3d(self):
        """Compute geometry for 3D quadrilateral panels."""
        num_panels = self.num_panels
        
        # Initialize arrays
        self.centers = np.zeros((num_panels, 3), dtype=np.float64)
        self.normals = np.zeros((num_panels, 3), dtype=np.float64)
        self.tangents = np.zeros((num_panels, 3), dtype=np.float64)
        self.areas = np.zeros(num_panels, dtype=np.float64)
        
        for i in range(num_panels):
            # Get panel corner nodes (assume CCW ordering)
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
            
            # Area (shoelace formula for quad, or split into triangles)
            # Using split triangle method: area = 0.5 * |d1 × d2|
            self.areas[i] = 0.5 * normal_mag
            
            # Tangent (first edge direction)
            tangent = p2 - p1
            tangent_mag = np.linalg.norm(tangent)
            if tangent_mag > 1e-14:
                self.tangents[i] = tangent / tangent_mag
            else:
                # Fallback: use second edge
                tangent = p3 - p2
                tangent_mag = np.linalg.norm(tangent)
                if tangent_mag > 1e-14:
                    self.tangents[i] = tangent / tangent_mag
                else:
                    # Degenerate panel
                    raise ValueError(f"Panel {i} has degenerate edges")
    
    def get_component_panels(self, component_id: int) -> NDArray[np.int32]:
        """
        Get panel indices belonging to a specific component.
        
        Args:
            component_id: Component identifier
        
        Returns:
            Array of panel indices
        """
        return np.where(self.component_ids == component_id)[0]
    
    def get_component_data(self, component_id: int, field_name: str) -> NDArray:
        """
        Extract field data for a specific component.
        
        Args:
            component_id: Component identifier
            field_name: Name of field in cell_data
        
        Returns:
            Field values for panels in this component
        """
        if field_name not in self.cell_data:
            raise KeyError(f"Field '{field_name}' not found in cell_data")
        
        panel_indices = self.get_component_panels(component_id)
        return self.cell_data[field_name][panel_indices]
    
    def __repr__(self) -> str:
        return (
            f"Mesh(dimension={self.dimension}, "
            f"nodes={self.num_nodes}, "
            f"panels={self.num_panels}, "
            f"components={len(np.unique(self.component_ids))})"
        )
