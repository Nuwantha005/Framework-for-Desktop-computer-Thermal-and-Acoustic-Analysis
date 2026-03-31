"""
Abstract base class for mesh data structures.

Defines the common interface for 2D and 3D panel meshes.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
from numpy.typing import NDArray


@dataclass
class MeshBase(ABC):
    """
    Abstract base class for 2D and 3D panel meshes.
    
    Provides the common interface that all mesh types must implement.
    Subclasses handle dimension-specific geometry computation.
    
    Attributes:
        nodes: Node coordinates (N, 3) - always 3D for consistency
        panels: Panel connectivity - (P, 2) for 2D lines, (P, 4) for 3D quads
        component_ids: Component ID for each panel (P,)
        centers: Panel center points (P, 3) - computed
        normals: Panel outward unit normals (P, 3) - computed
        areas: Panel lengths (2D) or areas (3D) (P,) - computed
        cell_data: Results storage dict, e.g., {'source_strength': array, 'Cp': array}
    """
    
    nodes: NDArray[np.float64]                    # (N, 3)
    panels: NDArray[np.int32]                     # (P, 2) or (P, 4)
    component_ids: NDArray[np.int32]              # (P,)
    
    # Computed geometry (set by compute_geometry())
    centers: Optional[NDArray[np.float64]] = None       # (P, 3)
    normals: Optional[NDArray[np.float64]] = None       # (P, 3)
    areas: Optional[NDArray[np.float64]] = None         # (P,)
    
    # Results storage
    cell_data: Dict[str, NDArray] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate mesh data and compute geometry."""
        self._validate()
        self.compute_geometry()
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return mesh dimension (2 or 3)."""
        pass
    
    @abstractmethod
    def compute_geometry(self) -> None:
        """Compute panel centers, normals, and areas."""
        pass
    
    def _validate(self) -> None:
        """Validate common mesh properties."""
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError(f"nodes must have shape (N, 3), got {self.nodes.shape}")
        
        if self.panels.ndim != 2:
            raise ValueError(f"panels must be 2D array, got {self.panels.ndim}D")
        
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
    def is_3d(self) -> bool:
        """Check if this is a 3D mesh."""
        return self.dimension == 3
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return self.nodes.shape[0]
    
    @property
    def num_panels(self) -> int:
        """Number of panels."""
        return self.panels.shape[0]
    
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
            f"{self.__class__.__name__}("
            f"nodes={self.num_nodes}, "
            f"panels={self.num_panels}, "
            f"components={len(np.unique(self.component_ids))})"
        )
