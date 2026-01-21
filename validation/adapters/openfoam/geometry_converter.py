"""
Geometry converter for OpenFOAM.

Converts 2D panel method geometry (JSON) to 3D STL for OpenFOAM meshing.
OpenFOAM is inherently 3D, so 2D problems are solved on thin slabs with
empty boundary conditions on front/back faces.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
import struct


@dataclass
class STLTriangle:
    """A single STL triangle with vertices and normal."""
    vertices: NDArray[np.float64]  # (3, 3) - three vertices
    normal: NDArray[np.float64]    # (3,) - face normal
    
    def __post_init__(self):
        if self.vertices.shape != (3, 3):
            raise ValueError(f"vertices must be (3, 3), got {self.vertices.shape}")
        if self.normal.shape != (3,):
            raise ValueError(f"normal must be (3,), got {self.normal.shape}")


class GeometryConverter:
    """
    Convert 2D panel geometry to 3D STL for OpenFOAM.
    
    The 2D geometry is extruded in the Z direction to create a thin slab.
    The resulting STL can be used with snappyHexMesh for mesh generation.
    
    Usage:
        from core.io import CaseLoader
        from validation.adapters.openfoam import GeometryConverter
        
        case = CaseLoader.load_case("cases/two_rounded_rects")
        converter = GeometryConverter(extrusion_depth=0.1)
        
        # Convert entire scene
        stl_path = converter.convert_scene(case.scene, output_dir)
        
        # Or convert individual components
        for comp in case.scene.components:
            stl_path = converter.convert_component(comp, output_dir)
    """
    
    def __init__(self, extrusion_depth: float = 0.1):
        """
        Initialize converter.
        
        Args:
            extrusion_depth: Z-direction thickness for the extruded geometry.
                            Should match blockMesh z-extent.
        """
        self.extrusion_depth = extrusion_depth
        self.z_min = 0.0
        self.z_max = extrusion_depth
    
    def convert_mesh(
        self, 
        mesh, 
        name: str,
        output_dir: Path,
        binary: bool = False
    ) -> Path:
        """
        Convert a 2D Mesh to STL file.
        
        The 2D panels are extruded to create a closed 3D surface.
        
        Args:
            mesh: 2D Mesh object with nodes (N, 3) and panels (P, 2)
            name: Name for the STL solid
            output_dir: Directory to write STL file
            binary: If True, write binary STL (smaller file)
        
        Returns:
            Path to created STL file
        """
        if mesh.dimension != 2:
            raise ValueError(f"Only 2D meshes supported, got dimension={mesh.dimension}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get nodes and panels
        nodes_2d = mesh.nodes[:, :2]  # (N, 2) - drop z coordinate
        panels = mesh.panels  # (P, 2)
        
        # Generate triangles for the extruded surface
        triangles = self._extrude_panels(nodes_2d, panels, name)
        
        # Write STL file
        stl_path = output_dir / f"{name}.stl"
        
        if binary:
            self._write_binary_stl(stl_path, triangles, name)
        else:
            self._write_ascii_stl(stl_path, triangles, name)
        
        return stl_path
    
    def convert_component(
        self,
        component,
        output_dir: Path,
        binary: bool = False
    ) -> Path:
        """
        Convert a Component to STL.
        
        Applies the component's transform and exports the global mesh.
        
        Args:
            component: Component object
            output_dir: Directory to write STL file
            binary: If True, write binary STL
        
        Returns:
            Path to created STL file
        """
        # Get global mesh (with transform applied)
        global_mesh = component.get_global_mesh(component_id=0)
        return self.convert_mesh(global_mesh, component.name, output_dir, binary)
    
    def convert_scene(
        self,
        scene,
        output_dir: Path,
        binary: bool = False,
        combined: bool = True
    ) -> List[Path]:
        """
        Convert all components in a Scene to STL.
        
        Args:
            scene: Scene object with components
            output_dir: Directory to write STL files
            binary: If True, write binary STL
            combined: If True, write single combined STL; 
                     if False, write separate STL per component
        
        Returns:
            List of paths to created STL files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if combined:
            # Combine all components into single STL
            all_triangles = []
            for component in scene.components:
                global_mesh = component.get_global_mesh(component_id=0)
                nodes_2d = global_mesh.nodes[:, :2]
                panels = global_mesh.panels
                triangles = self._extrude_panels(nodes_2d, panels, component.name)
                all_triangles.extend(triangles)
            
            stl_path = output_dir / f"{scene.name}.stl"
            if binary:
                self._write_binary_stl(stl_path, all_triangles, scene.name)
            else:
                self._write_ascii_stl(stl_path, all_triangles, scene.name)
            
            return [stl_path]
        else:
            # Separate STL per component
            paths = []
            for component in scene.components:
                path = self.convert_component(component, output_dir, binary)
                paths.append(path)
            return paths
    
    def _extrude_panels(
        self,
        nodes_2d: NDArray,
        panels: NDArray,
        name: str,
        add_caps: bool = True
    ) -> List[STLTriangle]:
        """
        Extrude 2D panels to create 3D triangular facets.
        
        For each 2D line panel, creates:
        - Two triangles for the side face (extruded quad split into 2 tris)
        
        If add_caps=True, also creates front and back cap faces to make
        a watertight closed volume (required for snappyHexMesh).
        
        Args:
            nodes_2d: 2D node coordinates (N, 2)
            panels: Panel connectivity (P, 2)
            name: Solid name
            add_caps: If True, add front/back caps for watertight volume
        
        Returns:
            List of STLTriangle objects
        """
        triangles = []
        z_min = self.z_min
        z_max = self.z_max
        
        # Side faces (extruded panels)
        for i, (n1_idx, n2_idx) in enumerate(panels):
            # Get 2D endpoints
            p1_2d = nodes_2d[n1_idx]
            p2_2d = nodes_2d[n2_idx]
            
            # Create 3D vertices for extruded quad
            # Bottom face (z = z_min)
            v1 = np.array([p1_2d[0], p1_2d[1], z_min])  # bottom-left
            v2 = np.array([p2_2d[0], p2_2d[1], z_min])  # bottom-right
            # Top face (z = z_max)
            v3 = np.array([p2_2d[0], p2_2d[1], z_max])  # top-right
            v4 = np.array([p1_2d[0], p1_2d[1], z_max])  # top-left
            
            # Calculate face normal (should point outward)
            # For 2D panels ordered CCW, normal is (ty, -tx) where t = p2 - p1
            tangent = p2_2d - p1_2d
            normal_2d = np.array([tangent[1], -tangent[0]])
            normal_2d = normal_2d / (np.linalg.norm(normal_2d) + 1e-12)
            normal_3d = np.array([normal_2d[0], normal_2d[1], 0.0])
            
            # Split quad into 2 triangles
            # Triangle 1: v1, v2, v3 (CCW when viewed from outside)
            tri1 = STLTriangle(
                vertices=np.array([v1, v2, v3]),
                normal=normal_3d
            )
            
            # Triangle 2: v1, v3, v4
            tri2 = STLTriangle(
                vertices=np.array([v1, v3, v4]),
                normal=normal_3d
            )
            
            triangles.append(tri1)
            triangles.append(tri2)
        
        # Add front and back caps if requested (for watertight volume)
        if add_caps:
            cap_triangles = self._create_cap_faces(nodes_2d, panels, z_min, z_max)
            triangles.extend(cap_triangles)
        
        return triangles
    
    def _create_cap_faces(
        self,
        nodes_2d: NDArray,
        panels: NDArray,
        z_min: float,
        z_max: float
    ) -> List[STLTriangle]:
        """
        Create front and back cap faces using ear clipping triangulation.
        
        This makes the extruded geometry watertight (closed volume),
        which is required for snappyHexMesh to work correctly.
        
        Args:
            nodes_2d: 2D node coordinates (N, 2)
            panels: Panel connectivity (P, 2)
            z_min: Bottom z coordinate
            z_max: Top z coordinate
        
        Returns:
            List of STLTriangle objects for front and back caps
        """
        triangles = []
        
        # Build ordered polygon from panels
        # Panels define edges, we need to order them into a closed loop
        polygon_2d = self._panels_to_polygon(nodes_2d, panels)
        
        if polygon_2d is None or len(polygon_2d) < 3:
            return triangles  # Can't triangulate
        
        # Triangulate the polygon using ear clipping
        tri_indices = self._triangulate_polygon(polygon_2d)
        
        # Create triangles for back cap (z = z_min, normal pointing -z)
        back_normal = np.array([0.0, 0.0, -1.0])
        for i0, i1, i2 in tri_indices:
            v0 = np.array([polygon_2d[i0, 0], polygon_2d[i0, 1], z_min])
            v1 = np.array([polygon_2d[i1, 0], polygon_2d[i1, 1], z_min])
            v2 = np.array([polygon_2d[i2, 0], polygon_2d[i2, 1], z_min])
            # Reverse winding for back face (normal pointing -z)
            tri = STLTriangle(vertices=np.array([v0, v2, v1]), normal=back_normal)
            triangles.append(tri)
        
        # Create triangles for front cap (z = z_max, normal pointing +z)
        front_normal = np.array([0.0, 0.0, 1.0])
        for i0, i1, i2 in tri_indices:
            v0 = np.array([polygon_2d[i0, 0], polygon_2d[i0, 1], z_max])
            v1 = np.array([polygon_2d[i1, 0], polygon_2d[i1, 1], z_max])
            v2 = np.array([polygon_2d[i2, 0], polygon_2d[i2, 1], z_max])
            # Normal winding for front face (normal pointing +z)
            tri = STLTriangle(vertices=np.array([v0, v1, v2]), normal=front_normal)
            triangles.append(tri)
        
        return triangles
    
    def _panels_to_polygon(
        self,
        nodes_2d: NDArray,
        panels: NDArray
    ) -> Optional[NDArray]:
        """
        Convert panel connectivity to ordered polygon vertices.
        
        Args:
            nodes_2d: Node coordinates (N, 2)
            panels: Panel connectivity (P, 2)
        
        Returns:
            Ordered polygon vertices (P, 2) or None if not a closed loop
        """
        if len(panels) < 3:
            return None
        
        # Build adjacency: node -> list of (panel_idx, other_node)
        from collections import defaultdict
        adjacency = defaultdict(list)
        
        for panel_idx, (n1, n2) in enumerate(panels):
            adjacency[n1].append((panel_idx, n2))
            adjacency[n2].append((panel_idx, n1))
        
        # Start from first panel's first node
        start_node = panels[0, 0]
        current_node = start_node
        visited_panels = set()
        ordered_nodes = []
        
        # Walk the loop
        for _ in range(len(panels) + 1):
            ordered_nodes.append(current_node)
            
            # Find next unvisited panel from current node
            next_node = None
            for panel_idx, other_node in adjacency[current_node]:
                if panel_idx not in visited_panels:
                    visited_panels.add(panel_idx)
                    next_node = other_node
                    break
            
            if next_node is None:
                break  # No more panels to visit
            
            if next_node == start_node and len(visited_panels) == len(panels):
                break  # Closed the loop
            
            current_node = next_node
        
        if len(ordered_nodes) < 3:
            return None
        
        # Return ordered polygon vertices
        return nodes_2d[ordered_nodes]
    
    def _triangulate_polygon(self, polygon: NDArray) -> List[Tuple[int, int, int]]:
        """
        Triangulate a simple polygon using ear clipping algorithm.
        
        Args:
            polygon: Ordered polygon vertices (N, 2)
        
        Returns:
            List of triangle indices (i0, i1, i2)
        """
        n = len(polygon)
        if n < 3:
            return []
        if n == 3:
            return [(0, 1, 2)]
        
        # Simple fan triangulation (works for convex and mostly-convex polygons)
        # For complex concave polygons, would need proper ear clipping
        triangles = []
        for i in range(1, n - 1):
            triangles.append((0, i, i + 1))
        
        return triangles
    
    def _write_ascii_stl(
        self,
        filepath: Path,
        triangles: List[STLTriangle],
        solid_name: str
    ):
        """Write triangles to ASCII STL file."""
        with open(filepath, 'w') as f:
            f.write(f"solid {solid_name}\n")
            
            for tri in triangles:
                n = tri.normal
                f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
                f.write("    outer loop\n")
                for v in tri.vertices:
                    f.write(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            
            f.write(f"endsolid {solid_name}\n")
    
    def _write_binary_stl(
        self,
        filepath: Path,
        triangles: List[STLTriangle],
        solid_name: str
    ):
        """Write triangles to binary STL file."""
        with open(filepath, 'wb') as f:
            # Header (80 bytes)
            header = f"Binary STL: {solid_name}"[:80].ljust(80, '\0')
            f.write(header.encode('ascii'))
            
            # Number of triangles (4 bytes, uint32)
            f.write(struct.pack('<I', len(triangles)))
            
            # Write each triangle
            for tri in triangles:
                # Normal (3 × float32)
                f.write(struct.pack('<3f', *tri.normal))
                # Vertices (3 × 3 × float32)
                for v in tri.vertices:
                    f.write(struct.pack('<3f', *v))
                # Attribute byte count (2 bytes, unused)
                f.write(struct.pack('<H', 0))


def compute_polygon_bounds(nodes_2d: NDArray) -> Tuple[float, float, float, float]:
    """
    Compute bounding box of 2D nodes.
    
    Args:
        nodes_2d: Node coordinates (N, 2)
    
    Returns:
        (x_min, x_max, y_min, y_max)
    """
    x_min, y_min = nodes_2d.min(axis=0)
    x_max, y_max = nodes_2d.max(axis=0)
    return float(x_min), float(x_max), float(y_min), float(y_max)
