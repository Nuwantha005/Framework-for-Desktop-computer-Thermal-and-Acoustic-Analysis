"""
Mesh format converters for validation.

Provides functions for converting panel method meshes to/from STL format
for use with OpenFOAM.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from core.geometry import Mesh, Component


def mesh_to_stl(
    mesh: Mesh,
    output_path: Path,
    extrude_depth: float = 0.1,
    component_name: str = "body"
) -> Path:
    """
    Convert a 2D panel mesh to a 3D watertight STL file.
    
    Extrudes the 2D mesh in the Z direction to create a 3D volume
    suitable for OpenFOAM meshing with snappyHexMesh.
    
    Args:
        mesh: Panel method mesh (2D)
        output_path: Path for output STL file
        extrude_depth: Depth of extrusion in Z direction
        component_name: Name for the component in STL header
    
    Returns:
        Path to created STL file
    
    Examples:
        >>> from core.geometry import Mesh
        >>> mesh = Mesh(...)  # 2D panel mesh
        >>> stl_path = mesh_to_stl(mesh, Path("output.stl"), extrude_depth=0.1)
        >>> print(f"Created: {stl_path}")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract 2D nodes (z=0 plane)
    nodes_2d = mesh.nodes[:, :2]
    num_nodes = len(nodes_2d)
    
    # Create extruded 3D nodes
    # Bottom face (z=0)
    bottom_nodes = np.column_stack([nodes_2d, np.zeros(num_nodes)])
    # Top face (z=extrude_depth)
    top_nodes = np.column_stack([nodes_2d, np.full(num_nodes, extrude_depth)])
    
    # Build triangles
    triangles = []
    
    # Front and back faces (need triangulation of polygon)
    # For now, use simple fan triangulation from first node
    # (works for convex shapes; complex shapes need proper triangulation)
    num_panels = mesh.num_panels
    
    # Front face (z=0, counter-clockwise when viewed from +z)
    for i in range(1, num_panels - 1):
        triangles.append([bottom_nodes[0], bottom_nodes[i], bottom_nodes[i+1]])
    
    # Back face (z=extrude_depth, clockwise when viewed from +z)
    for i in range(1, num_panels - 1):
        triangles.append([top_nodes[0], top_nodes[i+1], top_nodes[i]])
    
    # Side faces (one quad per panel, split into 2 triangles)
    for i in range(num_panels):
        i_next = (i + 1) % num_panels
        
        # Bottom-left, top-left, top-right
        triangles.append([
            bottom_nodes[i],
            top_nodes[i],
            top_nodes[i_next]
        ])
        
        # Bottom-left, top-right, bottom-right
        triangles.append([
            bottom_nodes[i],
            top_nodes[i_next],
            bottom_nodes[i_next]
        ])
    
    # Write ASCII STL
    with open(output_path, 'w') as f:
        f.write(f"solid {component_name}\n")
        
        for tri in triangles:
            # Compute normal (right-hand rule)
            v1 = tri[1] - tri[0]
            v2 = tri[2] - tri[0]
            normal = np.cross(v1, v2)
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            
            f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            f.write(f"    outer loop\n")
            for vertex in tri:
                f.write(f"      vertex {vertex[0]:.6e} {vertex[1]:.6e} {vertex[2]:.6e}\n")
            f.write(f"    endloop\n")
            f.write(f"  endfacet\n")
        
        f.write(f"endsolid {component_name}\n")
    
    return output_path


def component_to_stl(
    component: Component,
    output_path: Path,
    extrude_depth: float = 0.1
) -> Path:
    """
    Convert a Component to STL (applies transform).
    
    Args:
        component: Component with local mesh and transform
        output_path: Path for output STL file
        extrude_depth: Depth of extrusion in Z direction
    
    Returns:
        Path to created STL file
    
    Examples:
        >>> from core.geometry import Component
        >>> component = Component(...)
        >>> stl_path = component_to_stl(component, Path("body.stl"))
    """
    # Get transformed mesh
    world_mesh = component.get_world_mesh()
    
    return mesh_to_stl(
        world_mesh,
        output_path,
        extrude_depth=extrude_depth,
        component_name=component.name
    )


def scene_to_stl_files(
    scene,
    output_dir: Path,
    extrude_depth: float = 0.1
) -> list[Path]:
    """
    Convert all components in a scene to separate STL files.
    
    Args:
        scene: Scene with multiple components
        output_dir: Directory for STL files
        extrude_depth: Depth of extrusion
    
    Returns:
        List of paths to created STL files
    
    Examples:
        >>> from core.geometry import Scene
        >>> scene = Scene(...)
        >>> stl_files = scene_to_stl_files(scene, Path("constant/triSurface"))
        >>> for path in stl_files:
        ...     print(f"Created: {path.name}")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stl_files = []
    
    for component in scene.components:
        output_path = output_dir / f"{component.name}.stl"
        path = component_to_stl(component, output_path, extrude_depth)
        stl_files.append(path)
    
    return stl_files


def stl_to_mesh(stl_path: Path) -> Mesh:
    """
    Load an STL file and extract 2D panel mesh from midplane.
    
    NOTE: This is a placeholder. Full implementation would require:
    - Reading STL file format
    - Detecting midplane nodes
    - Reconstructing panel connectivity
    - Handling coordinate transformations
    
    Args:
        stl_path: Path to STL file
    
    Returns:
        Extracted 2D panel mesh
    
    Examples:
        >>> mesh = stl_to_mesh(Path("body.stl"))
    """
    raise NotImplementedError(
        "STL to Mesh conversion not yet implemented. "
        "Use mesh_to_stl for forward conversion only."
    )


def read_stl_triangles(stl_path: Path) -> tuple[NDArray, NDArray]:
    """
    Read triangles and normals from an ASCII STL file.
    
    Args:
        stl_path: Path to ASCII STL file
    
    Returns:
        (vertices, normals) where vertices is (N, 3, 3) and normals is (N, 3)
    
    Examples:
        >>> vertices, normals = read_stl_triangles(Path("body.stl"))
        >>> print(f"Triangles: {len(vertices)}")
    """
    stl_path = Path(stl_path)
    
    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")
    
    vertices = []
    normals = []
    current_normal = None
    current_triangle = []
    
    with open(stl_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('facet normal'):
                parts = line.split()
                current_normal = [float(parts[2]), float(parts[3]), float(parts[4])]
            
            elif line.startswith('vertex'):
                parts = line.split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                current_triangle.append(vertex)
            
            elif line.startswith('endfacet'):
                if len(current_triangle) == 3:
                    vertices.append(current_triangle)
                    normals.append(current_normal)
                current_triangle = []
                current_normal = None
    
    return np.array(vertices), np.array(normals)


def validate_stl(stl_path: Path, verbose: bool = True) -> bool:
    """
    Validate an STL file for basic correctness.
    
    Args:
        stl_path: Path to STL file
        verbose: Print validation messages
    
    Returns:
        True if valid
    
    Examples:
        >>> is_valid = validate_stl(Path("body.stl"))
        >>> if not is_valid:
        ...     print("STL file has issues")
    """
    try:
        vertices, normals = read_stl_triangles(stl_path)
        
        if len(vertices) == 0:
            if verbose:
                print(f"ERROR: No triangles found in {stl_path}")
            return False
        
        if verbose:
            print(f"STL Validation: {stl_path.name}")
            print(f"  Triangles: {len(vertices)}")
            print(f"  Bounds: X=[{vertices[:,:,0].min():.3f}, {vertices[:,:,0].max():.3f}]")
            print(f"          Y=[{vertices[:,:,1].min():.3f}, {vertices[:,:,1].max():.3f}]")
            print(f"          Z=[{vertices[:,:,2].min():.3f}, {vertices[:,:,2].max():.3f}]")
        
        # Check for degenerate triangles
        areas = []
        for tri in vertices:
            v1 = tri[1] - tri[0]
            v2 = tri[2] - tri[0]
            area = np.linalg.norm(np.cross(v1, v2)) / 2
            areas.append(area)
        
        min_area = np.min(areas)
        if min_area < 1e-12:
            if verbose:
                print(f"  WARNING: Found degenerate triangles (min area={min_area:.2e})")
        
        if verbose:
            print("  ✓ Valid STL file")
        
        return True
    
    except Exception as e:
        if verbose:
            print(f"ERROR validating STL: {e}")
        return False


__all__ = [
    'mesh_to_stl',
    'component_to_stl',
    'scene_to_stl_files',
    'stl_to_mesh',
    'read_stl_triangles',
    'validate_stl',
]
