"""
Geometry file readers for JSON and XY formats.
"""

from pathlib import Path
from typing import Dict, Any
import json
import numpy as np
from numpy.typing import NDArray

from ..geometry.mesh import Mesh


class GeometryReader:
    """Base class for geometry readers."""
    
    @staticmethod
    def read_json(filepath: str | Path) -> Mesh:
        """
        Read geometry from JSON file.
        
        Expected format:
        {
          "format": "panels_2d" or "panels_3d",
          "nodes": [[x, y, z], ...],
          "panels": [[i1, i2], ...] for 2D or [[i1, i2, i3, i4], ...] for 3D,
          "normal_direction": "outward" or "inward" (optional)
        }
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            Mesh object
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Geometry file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Parse format
        fmt = data.get("format", "").lower()
        if fmt not in ("panels_2d", "panels_3d"):
            raise ValueError(
                f"Unsupported format '{fmt}'. Expected 'panels_2d' or 'panels_3d'"
            )
        
        dimension = 2 if fmt == "panels_2d" else 3
        
        # Parse nodes
        nodes_list = data.get("nodes")
        if nodes_list is None:
            raise ValueError("Missing 'nodes' field in JSON")
        
        nodes = np.array(nodes_list, dtype=np.float64)
        
        # Ensure nodes are 3D
        if nodes.ndim != 2:
            raise ValueError(f"nodes must be 2D array, got {nodes.ndim}D")
        
        if nodes.shape[1] == 2:
            # Pad 2D coordinates with z=0
            nodes = np.hstack([nodes, np.zeros((nodes.shape[0], 1))])
        elif nodes.shape[1] != 3:
            raise ValueError(f"nodes must have 2 or 3 columns, got {nodes.shape[1]}")
        
        # Parse panels
        panels_list = data.get("panels")
        if panels_list is None:
            raise ValueError("Missing 'panels' field in JSON")
        
        panels = np.array(panels_list, dtype=np.int32)
        
        if panels.ndim != 2:
            raise ValueError(f"panels must be 2D array, got {panels.ndim}D")
        
        expected_panel_size = 2 if dimension == 2 else 4
        if panels.shape[1] != expected_panel_size:
            raise ValueError(
                f"For {fmt}, panels must have {expected_panel_size} columns, "
                f"got {panels.shape[1]}"
            )
        
        # Check normal direction (optional)
        normal_dir = data.get("normal_direction", "outward").lower()
        if normal_dir == "inward":
            # Reverse panel connectivity to flip normals
            panels = np.flip(panels, axis=1)
        
        # Create mesh (all panels initially belong to component 0)
        component_ids = np.zeros(panels.shape[0], dtype=np.int32)
        
        mesh = Mesh(
            nodes=nodes,
            panels=panels,
            dimension=dimension,
            component_ids=component_ids
        )
        
        return mesh
    
    @staticmethod
    def read_xy(filepath: str | Path) -> Mesh:
        """
        Read 2D point cloud from XY file and generate line panels.
        
        Expected format (space or tab separated):
        # Optional comment lines
        x1 y1 [z1]
        x2 y2 [z2]
        ...
        
        Panels are created connecting consecutive points (closed loop).
        
        Args:
            filepath: Path to XY file
        
        Returns:
            Mesh object (2D)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Geometry file not found: {filepath}")
        
        # Read data, skip comment lines
        points = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2]) if len(parts) >= 3 else 0.0
                    points.append([x, y, z])
        
        if len(points) < 2:
            raise ValueError(f"XY file must contain at least 2 points, got {len(points)}")
        
        nodes = np.array(points, dtype=np.float64)
        
        # Generate panels (connect consecutive points, close loop)
        num_points = len(points)
        panels = np.array(
            [[i, (i + 1) % num_points] for i in range(num_points)],
            dtype=np.int32
        )
        
        component_ids = np.zeros(panels.shape[0], dtype=np.int32)
        
        mesh = Mesh(
            nodes=nodes,
            panels=panels,
            dimension=2,
            component_ids=component_ids
        )
        
        return mesh
    
    @staticmethod
    def read(filepath: str | Path) -> Mesh:
        """
        Auto-detect format and read geometry file.
        
        Args:
            filepath: Path to geometry file
        
        Returns:
            Mesh object
        """
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()
        
        if suffix == '.json':
            return GeometryReader.read_json(filepath)
        elif suffix in ('.xy', '.dat', '.txt'):
            return GeometryReader.read_xy(filepath)
        else:
            raise ValueError(
                f"Unsupported file extension '{suffix}'. "
                f"Supported: .json, .xy, .dat, .txt"
            )


def generate_rectangle(width: float, height: float, 
                       center: tuple[float, float] = (0.0, 0.0),
                       num_panels_x: int = None,
                       num_panels_y: int = None) -> Mesh:
    """
    Generate rectangular mesh in 2D.
    
    Args:
        width: Rectangle width
        height: Rectangle height
        center: Center point (x, y)
        num_panels_x: Number of panels on horizontal sides (default: 1)
        num_panels_y: Number of panels on vertical sides (default: 1)
    
    Returns:
        2D Mesh of rectangle
    """
    if num_panels_x is None:
        num_panels_x = 1
    if num_panels_y is None:
        num_panels_y = 1
    
    cx, cy = center
    x0, x1 = cx - width / 2, cx + width / 2
    y0, y1 = cy - height / 2, cy + height / 2
    
    # Generate nodes around perimeter
    # Bottom edge (y0, left to right)
    bottom_x = np.linspace(x0, x1, num_panels_x + 1)
    bottom_nodes = np.column_stack([bottom_x, np.full_like(bottom_x, y0), np.zeros_like(bottom_x)])
    
    # Right edge (x1, bottom to top, exclude first point to avoid duplicate)
    right_y = np.linspace(y0, y1, num_panels_y + 1)[1:]
    right_nodes = np.column_stack([np.full_like(right_y, x1), right_y, np.zeros_like(right_y)])
    
    # Top edge (y1, right to left, exclude first)
    top_x = np.linspace(x1, x0, num_panels_x + 1)[1:]
    top_nodes = np.column_stack([top_x, np.full_like(top_x, y1), np.zeros_like(top_x)])
    
    # Left edge (x0, top to bottom, exclude first and last to close loop)
    left_y = np.linspace(y1, y0, num_panels_y + 1)[1:-1]
    left_nodes = np.column_stack([np.full_like(left_y, x0), left_y, np.zeros_like(left_y)])
    
    # Concatenate all nodes
    nodes = np.vstack([bottom_nodes, right_nodes, top_nodes, left_nodes])
    
    # Generate panels
    num_nodes = nodes.shape[0]
    panels = np.array([[i, (i + 1) % num_nodes] for i in range(num_nodes)], dtype=np.int32)
    
    component_ids = np.zeros(panels.shape[0], dtype=np.int32)
    
    return Mesh(nodes=nodes, panels=panels, dimension=2, component_ids=component_ids)


def generate_circle(radius: float, num_panels: int,
                    center: tuple[float, float] = (0.0, 0.0)) -> Mesh:
    """
    Generate circular mesh in 2D.
    
    Args:
        radius: Circle radius
        num_panels: Number of panels around circumference
        center: Center point (x, y)
    
    Returns:
        2D Mesh of circle
    """
    cx, cy = center
    
    # Generate points around circle (CCW from +x axis)
    theta = np.linspace(0, 2 * np.pi, num_panels + 1)[:-1]  # Exclude duplicate at 2π
    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)
    z = np.zeros_like(x)
    
    nodes = np.column_stack([x, y, z])
    
    # Generate panels
    panels = np.array([[i, (i + 1) % num_panels] for i in range(num_panels)], dtype=np.int32)
    
    component_ids = np.zeros(panels.shape[0], dtype=np.int32)
    
    return Mesh(nodes=nodes, panels=panels, dimension=2, component_ids=component_ids)
