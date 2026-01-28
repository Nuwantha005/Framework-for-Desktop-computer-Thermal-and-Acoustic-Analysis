"""
Geometry Mapper for Arc Length Projection

Projects simulation data points onto a high-resolution reference geometry to:
1. Calculate consistent arc lengths across different mesh resolutions
2. Eliminate offset/phase shift errors between datasets
3. Handle arbitrary geometries (convex, non-convex, etc.)

This is the standard engineering approach for comparing CFD results on different meshes.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional


class GeometryMapper:
    """
    Maps simulation points to arc length coordinates along a reference geometry.
    
    Uses vectorized projection to find the closest point on a reference polyline
    (master curve) and computes accurate arc length coordinates.
    
    Attributes:
        nodes: Reference geometry nodes (N, 2) in 2D
        total_length: Total perimeter/arc length of reference geometry
        node_arc_lengths: Cumulative arc length at each reference node
    """
    
    def __init__(self, reference_nodes: NDArray):
        """
        Initialize mapper with high-resolution reference geometry.
        
        Args:
            reference_nodes: (N, 2) or (N, 3) array of ordered points defining
                           the exact geometry. Should be high resolution 
                           (e.g., from STL with fine tessellation).
        """
        # Keep only 2D coordinates (x, y)
        self.nodes = reference_nodes[:, :2].astype(np.float64)
        
        # Calculate segment vectors and lengths
        # Segments go from nodes[i] to nodes[i+1]
        self.diffs = np.diff(self.nodes, axis=0)  # Vectors (dx, dy)
        self.seg_lengths = np.linalg.norm(self.diffs, axis=1)  # Length of each segment
        self.seg_lengths_sq = self.seg_lengths ** 2
        
        # Avoid division by zero for degenerate segments
        self.seg_lengths_sq[self.seg_lengths_sq == 0] = 1e-12
        
        # Calculate cumulative arc length at each node
        self.node_arc_lengths = np.concatenate(([0], np.cumsum(self.seg_lengths)))
        self.total_length = self.node_arc_lengths[-1]
    
    def get_arc_length(self, query_points: NDArray) -> NDArray:
        """
        Project query points onto reference geometry and return arc lengths.
        
        For each query point, finds the closest point on the reference polyline
        and computes the arc length to that projection point.
        
        Args:
            query_points: (M, 2) or (M, 3) array of points to map
        
        Returns:
            Array of arc lengths (M,) for each query point
        """
        query_points = query_points[:, :2].astype(np.float64)
        
        A = self.nodes[:-1]  # Segment starts
        V = self.diffs       # Segment vectors (B - A)
        
        s_results = []
        
        # Process each query point
        for P in query_points:
            # Vector from segment start to query point: W = P - A
            W = P - A  # Shape: (num_segments, 2)
            
            # Project W onto V: t = dot(W, V) / dot(V, V)
            # t represents the fraction along each segment [0, 1]
            c1 = np.sum(W * V, axis=1)
            t = c1 / self.seg_lengths_sq
            
            # Clamp t to segment bounds [0, 1]
            t_clipped = np.clip(t, 0, 1)
            
            # Find closest point on each segment: closest = A + t * V
            projections = A + t_clipped[:, np.newaxis] * V
            
            # Distance from query point to closest point on each segment
            dists = np.linalg.norm(P - projections, axis=1)
            
            # Find segment with minimum distance
            best_idx = np.argmin(dists)
            
            # Calculate arc length: distance at segment start + fraction of segment
            s = self.node_arc_lengths[best_idx] + t_clipped[best_idx] * self.seg_lengths[best_idx]
            s_results.append(s)
        
        return np.array(s_results)
    
    def normalize_arc_length(
        self, 
        arc_lengths: NDArray, 
        reference_point: Optional[NDArray] = None,
        landmark: str = "min_x"
    ) -> NDArray:
        """
        Normalize arc lengths relative to a geometric landmark.
        
        This eliminates offset/phase shift between different datasets by
        setting S=0 at a consistent physical location.
        
        Args:
            arc_lengths: Arc length values to normalize
            reference_point: Specific (x, y) point to use as S=0. If None, uses landmark.
            landmark: Geometric feature to use as S=0. Options:
                - "min_x": Leftmost point (leading edge for horizontal flow)
                - "max_x": Rightmost point
                - "min_y": Bottommost point
                - "max_y": Topmost point
        
        Returns:
            Normalized arc lengths (S=0 at landmark, wraps at total_length)
        """
        if reference_point is not None:
            # Find arc length of the closest node to reference point
            dists = np.linalg.norm(self.nodes - reference_point[:2], axis=1)
            ref_idx = np.argmin(dists)
            s_shift = self.node_arc_lengths[ref_idx]
        else:
            # Use geometric landmark
            if landmark == "min_x":
                ref_idx = np.argmin(self.nodes[:, 0])
            elif landmark == "max_x":
                ref_idx = np.argmax(self.nodes[:, 0])
            elif landmark == "min_y":
                ref_idx = np.argmin(self.nodes[:, 1])
            elif landmark == "max_y":
                ref_idx = np.argmax(self.nodes[:, 1])
            else:
                raise ValueError(f"Unknown landmark: {landmark}")
            
            s_shift = self.node_arc_lengths[ref_idx]
        
        # Shift and wrap around total length
        normalized = (arc_lengths - s_shift) % self.total_length
        
        return normalized
    
    @classmethod
    def from_stl(cls, stl_path: str, component_name: Optional[str] = None) -> 'GeometryMapper':
        """
        Create mapper from STL file.
        
        STL files contain triangulated surfaces. We extract the boundary edges
        and order them to form the reference polyline.
        
        Args:
            stl_path: Path to STL file
            component_name: Optional component name for multi-component STL
        
        Returns:
            GeometryMapper instance
        """
        import pyvista as pv
        
        # Read STL
        mesh = pv.read(stl_path)
        
        # Extract boundary edges
        # For 2D extrusions (thin slices), extract one of the z=constant faces
        # This is simpler: just take all unique points and sort them
        points = mesh.points
        
        # For 2D case, find points at z=0 plane (or min z)
        z_min = points[:, 2].min()
        z_mask = np.abs(points[:, 2] - z_min) < 1e-6
        boundary_points = points[z_mask, :2]
        
        # Remove duplicates
        boundary_points = np.unique(boundary_points, axis=0)
        
        # Sort by angle from centroid (for closed curves)
        cx = np.mean(boundary_points[:, 0])
        cy = np.mean(boundary_points[:, 1])
        angles = np.arctan2(boundary_points[:, 1] - cy, boundary_points[:, 0] - cx)
        sort_idx = np.argsort(angles)
        ordered_points = boundary_points[sort_idx]
        
        return cls(ordered_points)
