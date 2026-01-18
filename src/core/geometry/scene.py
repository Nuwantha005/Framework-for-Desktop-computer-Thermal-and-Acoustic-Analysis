"""
Scene class for managing multiple components and assembling global mesh.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np
from numpy.typing import NDArray

from .component import Component
from .mesh import Mesh


@dataclass
class Scene:
    """
    Collection of components forming the complete problem domain.
    
    Attributes:
        name: Scene identifier
        components: List of Component objects
        freestream: Freestream velocity vector (3,)
        description: Optional scene description
    """
    
    name: str
    components: List[Component]
    freestream: NDArray[np.float64]
    description: str = ""
    
    def __post_init__(self):
        """Validate scene data."""
        if self.freestream.shape != (3,):
            raise ValueError(f"freestream must have shape (3,), got {self.freestream.shape}")
        
        if len(self.components) == 0:
            raise ValueError("Scene must have at least one component")
        
        # Check for duplicate names
        names = [comp.name for comp in self.components]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Duplicate component names: {set(duplicates)}")
    
    @property
    def num_components(self) -> int:
        """Number of components in scene."""
        return len(self.components)
    
    def assemble(self) -> Mesh:
        """
        Assemble all components into a single global mesh.
        
        Each component is assigned a unique ID (0, 1, 2, ...).
        All local meshes are transformed to global coordinates and concatenated.
        
        Returns:
            Global mesh containing all components
        """
        if len(self.components) == 0:
            raise ValueError("Cannot assemble empty scene")
        
        # Check dimension consistency
        dimension = self.components[0].local_mesh.dimension
        for comp in self.components[1:]:
            if comp.local_mesh.dimension != dimension:
                raise ValueError(
                    f"All components must have same dimension. "
                    f"Component '{comp.name}' has dimension {comp.local_mesh.dimension}, "
                    f"expected {dimension}"
                )
        
        # Transform each component to global coordinates
        global_meshes = []
        for component_id, component in enumerate(self.components):
            global_mesh = component.get_global_mesh(component_id)
            global_meshes.append(global_mesh)
        
        # Concatenate all meshes
        return self._concatenate_meshes(global_meshes, dimension)
    
    def _concatenate_meshes(self, meshes: List[Mesh], dimension: int) -> Mesh:
        """
        Concatenate multiple meshes into one.
        
        Args:
            meshes: List of meshes to concatenate
            dimension: Problem dimension
        
        Returns:
            Combined mesh
        """
        # Concatenate nodes
        all_nodes = []
        all_panels = []
        all_component_ids = []
        all_centers = []
        all_normals = []
        all_tangents = []
        all_areas = []
        
        node_offset = 0
        
        for mesh in meshes:
            # Nodes
            all_nodes.append(mesh.nodes)
            
            # Panels (shift node indices by offset)
            shifted_panels = mesh.panels + node_offset
            all_panels.append(shifted_panels)
            
            # Component IDs
            all_component_ids.append(mesh.component_ids)
            
            # Geometry
            if mesh.centers is not None:
                all_centers.append(mesh.centers)
            if mesh.normals is not None:
                all_normals.append(mesh.normals)
            if mesh.tangents is not None:
                all_tangents.append(mesh.tangents)
            if mesh.areas is not None:
                all_areas.append(mesh.areas)
            
            node_offset += mesh.num_nodes
        
        # Concatenate arrays
        global_nodes = np.vstack(all_nodes)
        global_panels = np.vstack(all_panels).astype(np.int32)
        global_component_ids = np.hstack(all_component_ids).astype(np.int32)
        
        # Create global mesh
        global_mesh = Mesh(
            nodes=global_nodes,
            panels=global_panels,
            dimension=dimension,
            component_ids=global_component_ids
        )
        
        # Override computed geometry (already transformed)
        if all_centers:
            global_mesh.centers = np.vstack(all_centers)
        if all_normals:
            global_mesh.normals = np.vstack(all_normals)
        if all_tangents:
            global_mesh.tangents = np.vstack(all_tangents)
        if all_areas:
            global_mesh.areas = np.hstack(all_areas)
        
        return global_mesh
    
    def get_component(self, name: str) -> Component:
        """
        Get component by name.
        
        Args:
            name: Component name
        
        Returns:
            Component object
        
        Raises:
            KeyError: If component not found
        """
        for comp in self.components:
            if comp.name == name:
                return comp
        raise KeyError(f"Component '{name}' not found in scene")
    
    def get_component_by_id(self, component_id: int) -> Component:
        """
        Get component by ID.
        
        Args:
            component_id: Component index (0-based)
        
        Returns:
            Component object
        
        Raises:
            IndexError: If ID out of range
        """
        if component_id < 0 or component_id >= len(self.components):
            raise IndexError(
                f"Component ID {component_id} out of range [0, {len(self.components)-1}]"
            )
        return self.components[component_id]
    
    def __repr__(self) -> str:
        total_panels = sum(comp.local_mesh.num_panels for comp in self.components)
        return (
            f"Scene(name='{self.name}', "
            f"components={self.num_components}, "
            f"total_panels={total_panels})"
        )
