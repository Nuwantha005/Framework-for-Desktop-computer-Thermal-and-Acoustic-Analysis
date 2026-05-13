"""
Gmsh-based geometry reader for STEP, STL, and other CAD formats.
"""

from pathlib import Path
import numpy as np
import warnings

def read_with_gmsh(path: str | Path, component_id: int = 0, scale: float = 1.0):
    """
    Read a CAD file (STEP, STL, etc.) using Gmsh and generate a quad surface mesh.
    
    Args:
        path: Path to CAD file
        component_id: Component ID for the generated mesh
        scale: Scaling factor for coordinates
        
    Returns:
        Mesh3D object
    """
    try:
        import gmsh
    except ImportError:
        raise ImportError("gmsh is required: pip install gmsh")
        
    path_str = str(path)
    
    gmsh.initialize()
    # Suppress output to terminal unless it's a warning/error
    gmsh.option.setNumber("General.Terminal", 0)
    
    try:
        gmsh.merge(path_str)
        gmsh.model.occ.synchronize()
        
        # Set algorithm to front-delaunay for quads
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        
        # Generate 2D mesh
        gmsh.model.mesh.generate(2)
        
        # Extract nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        
        if len(node_tags) == 0:
            raise ValueError("Gmsh generated no nodes. Check the input file.")
            
        # Map node tags to 0-based indices
        tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}
        
        nodes = np.array(node_coords, dtype=np.float64).reshape(-1, 3)
        nodes *= scale
        
        # Extract 2D elements
        element_types, element_tags, node_tags_per_elem = gmsh.model.mesh.getElements(dim=2)
        
        panels_list = []
        for i, etype in enumerate(element_types):
            num_nodes = 0
            if etype == 2:  # 3-node triangle
                num_nodes = 3
                warnings.warn(f"Gmsh left some triangles in {path.name}. Padding to degenerate quads.")
            elif etype == 3:  # 4-node quadrangle
                num_nodes = 4
            else:
                continue # ignore other types
                
            tags = node_tags_per_elem[i].reshape(-1, num_nodes)
            for elem_tags in tags:
                indices = [tag_to_idx[t] for t in elem_tags]
                if num_nodes == 3:
                    # Pad to degenerate quad: [i, j, k, k]
                    indices.append(indices[-1])
                panels_list.append(indices)
                
        if not panels_list:
            raise ValueError("Gmsh generated no surface elements.")
            
        panels = np.array(panels_list, dtype=np.int32)
        component_ids = np.full(len(panels), component_id, dtype=np.int32)
        
        from ..mesh3d import Mesh3D
        return Mesh3D(
            nodes=nodes,
            panels=panels,
            component_ids=component_ids
        )
        
    finally:
        gmsh.finalize()
