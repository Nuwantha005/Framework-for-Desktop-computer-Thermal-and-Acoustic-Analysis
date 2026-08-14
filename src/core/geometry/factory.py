"""
Geometry factory for parametric mesh generation.

Registry-based factory that maps geometry type strings to generator functions.
Supports tuple-based resolution parameters that unpack directly to function arguments.
"""

from typing import Dict, Callable, Any, List

from .mesh_base import MeshBase
from . import generators
from .io import sphere_generator


class GeometryFactory:
    """
    Factory for creating parametric geometry at specified resolution levels.
    
    Uses registry pattern to map geometry type strings (from case.yaml) to 
    generator functions. Resolution tuples are unpacked as first arguments.
    
    Example:
        >>> factory = GeometryFactory()
        >>> geom_def = {"type": "rectangle", "parameters": {"width": 1.0, "height": 1.0}}
        >>> mesh = factory.create(geom_def, resolution=[4, 4])
        >>> mesh.num_panels
        16
    """
    
    # Registry mapping type strings to generator functions
    _generators: Dict[str, Callable[..., MeshBase]] = {
        "circle": generators.generate_circle,
        "rectangle": generators.generate_rectangle,
        "rounded_rectangle": generators.generate_rounded_rectangle,
        "sphere": sphere_generator.generate_sphere,
        "cylinder": sphere_generator.generate_cylinder,
        "thick_cylinder": sphere_generator.generate_thick_cylinder,
    }
    
    @classmethod
    def create(cls, geometry_definition: dict, resolution: List[int]) -> MeshBase:
        """
        Create geometry mesh at specified resolution.
        
        Args:
            geometry_definition: Dict with 'type' and 'parameters' keys
                - type: Geometry type string (e.g., "rectangle", "circle")
                - parameters: Dict of shape parameters (width, height, radius, etc.)
            resolution: List of resolution values that unpack to generator's first args
                - For circle: [num_panels]
                - For rectangle: [num_panels_x, num_panels_y]
                - For rounded_rectangle: [num_panels_per_side, num_panels_per_arc]
        
        Returns:
            Generated Mesh object
        
        Raises:
            ValueError: If geometry type not registered or parameters invalid
            TypeError: If resolution tuple length doesn't match function signature
        
        Example:
            >>> # Circle with 32 panels
            >>> geom = {"type": "circle", "parameters": {"radius": 0.5}}
            >>> mesh = GeometryFactory.create(geom, [32])
            
            >>> # Rectangle with 8x8 panels
            >>> geom = {"type": "rectangle", "parameters": {"width": 1.0, "height": 1.0}}
            >>> mesh = GeometryFactory.create(geom, [8, 8])
        """
        geom_type = geometry_definition.get("type")
        if geom_type not in cls._generators:
            available = ", ".join(cls._generators.keys())
            raise ValueError(
                f"Unknown geometry type '{geom_type}'. "
                f"Available types: {available}"
            )
        
        generator = cls._generators[geom_type]
        params = geometry_definition.get("parameters", {})
        
        try:
            # Unpack resolution tuple as first arguments, then pass shape parameters
            mesh = generator(*resolution, **params)
        except TypeError as e:
            raise TypeError(
                f"Invalid resolution tuple {resolution} for geometry type '{geom_type}'. "
                f"Check generator function signature. Original error: {e}"
            ) from e
        
        return mesh
    
    @classmethod
    def register(cls, type_name: str, generator_func: Callable[..., MeshBase]) -> None:
        """
        Register custom geometry generator.
        
        Args:
            type_name: String identifier for geometry type
            generator_func: Function that takes (*resolution_params, **shape_params) and returns MeshBase
        
        Example:
            >>> def generate_ellipse(num_panels, a, b, center=(0, 0)):
            ...     # ... implementation
            ...     return Mesh(...)
            >>> GeometryFactory.register("ellipse", generate_ellipse)
        """
        if type_name in cls._generators:
            raise ValueError(f"Geometry type '{type_name}' already registered")
        cls._generators[type_name] = generator_func
    
    @classmethod
    def list_types(cls) -> List[str]:
        """Return list of registered geometry types."""
        return list(cls._generators.keys())
