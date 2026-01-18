"""
Test geometry foundation: mesh, transforms, scene assembly.
"""

import pytest
import numpy as np
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.geometry import Point3D, Vector3D, Mesh, Transform, Component, Scene
from core.io import GeometryReader, generate_rectangle, generate_circle, CaseLoader


class TestPrimitives:
    """Test Point3D and Vector3D."""
    
    def test_point_creation(self):
        p = Point3D(1.0, 2.0, 3.0)
        assert p.x == 1.0
        assert p.y == 2.0
        assert p.z == 3.0
    
    def test_point_to_array(self):
        p = Point3D(1.0, 2.0, 3.0)
        arr = p.to_array()
        np.testing.assert_array_almost_equal(arr, [1.0, 2.0, 3.0])
    
    def test_vector_operations(self):
        v1 = Vector3D(1.0, 0.0, 0.0)
        v2 = Vector3D(0.0, 1.0, 0.0)
        
        # Dot product
        assert v1.dot(v2) == 0.0
        
        # Cross product (should give z-axis)
        v3 = v1.cross(v2)
        np.testing.assert_array_almost_equal(v3.to_array(), [0.0, 0.0, 1.0])
        
        # Magnitude
        assert v1.magnitude() == 1.0
        
        # Normalize
        v4 = Vector3D(3.0, 4.0, 0.0)
        v4_norm = v4.normalize()
        assert abs(v4_norm.magnitude() - 1.0) < 1e-10


class TestMesh:
    """Test Mesh class."""
    
    def test_square_mesh(self):
        # Create unit square
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        panels = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
        component_ids = np.zeros(4, dtype=np.int32)
        
        mesh = Mesh(nodes=nodes, panels=panels, dimension=2, component_ids=component_ids)
        
        assert mesh.num_nodes == 4
        assert mesh.num_panels == 4
        assert mesh.is_2d
        
        # Check computed geometry
        assert mesh.centers is not None
        assert mesh.normals is not None
        assert mesh.areas is not None
        
        # Panel 0 (bottom edge): normal should point down (0, -1, 0)
        np.testing.assert_array_almost_equal(mesh.normals[0], [0.0, -1.0, 0.0])
        
        # Panel 0 length should be 1.0
        assert abs(mesh.areas[0] - 1.0) < 1e-10
    
    def test_generate_rectangle(self):
        mesh = generate_rectangle(width=2.0, height=1.0, center=(0.0, 0.0))
        
        assert mesh.is_2d
        assert mesh.num_panels == 4  # Default 1 panel per side
        
        # Check bounds
        x_coords = mesh.nodes[:, 0]
        y_coords = mesh.nodes[:, 1]
        
        assert abs(x_coords.max() - 1.0) < 1e-10
        assert abs(x_coords.min() + 1.0) < 1e-10
        assert abs(y_coords.max() - 0.5) < 1e-10
        assert abs(y_coords.min() + 0.5) < 1e-10
    
    def test_generate_circle(self):
        mesh = generate_circle(radius=1.0, num_panels=64)
        
        assert mesh.is_2d
        assert mesh.num_panels == 64
        
        # Check all nodes are approximately distance 1.0 from origin
        distances = np.linalg.norm(mesh.nodes[:, :2], axis=1)
        np.testing.assert_array_almost_equal(distances, np.ones(64), decimal=10)


class TestTransform:
    """Test Transform class."""
    
    def test_identity_transform(self):
        t = Transform.identity()
        
        point = np.array([1.0, 2.0, 3.0])
        transformed = t.apply_to_point(point)
        
        np.testing.assert_array_almost_equal(transformed, point)
    
    def test_2d_translation(self):
        t = Transform.from_2d(tx=3.0, ty=4.0, angle_deg=0.0)
        
        point = np.array([1.0, 2.0, 0.0])
        transformed = t.apply_to_point(point)
        
        np.testing.assert_array_almost_equal(transformed, [4.0, 6.0, 0.0])
    
    def test_2d_rotation(self):
        # 90 degree rotation CCW
        t = Transform.from_2d(tx=0.0, ty=0.0, angle_deg=90.0)
        
        point = np.array([1.0, 0.0, 0.0])
        transformed = t.apply_to_point(point)
        
        # Should rotate to (0, 1, 0)
        np.testing.assert_array_almost_equal(transformed, [0.0, 1.0, 0.0], decimal=10)


class TestComponent:
    """Test Component class."""
    
    def test_component_transform(self):
        # Create unit square mesh
        local_mesh = generate_rectangle(width=1.0, height=1.0, center=(0.0, 0.0))
        
        # Transform: translate by (2, 3)
        transform = Transform.from_2d(tx=2.0, ty=3.0, angle_deg=0.0)
        
        component = Component(
            name="test_square",
            local_mesh=local_mesh,
            transform=transform,
            bc_type="wall"
        )
        
        global_mesh = component.get_global_mesh(component_id=0)
        
        # Check translation applied
        x_coords = global_mesh.nodes[:, 0]
        y_coords = global_mesh.nodes[:, 1]
        
        assert abs(x_coords.min() - 1.5) < 1e-10  # -0.5 + 2.0
        assert abs(x_coords.max() - 2.5) < 1e-10  # +0.5 + 2.0
        assert abs(y_coords.min() - 2.5) < 1e-10
        assert abs(y_coords.max() - 3.5) < 1e-10


class TestScene:
    """Test Scene assembly."""
    
    def test_single_component_scene(self):
        # Single square
        local_mesh = generate_rectangle(width=1.0, height=1.0)
        transform = Transform.identity()
        
        component = Component(
            name="square",
            local_mesh=local_mesh,
            transform=transform,
            bc_type="wall"
        )
        
        scene = Scene(
            name="single_square",
            components=[component],
            freestream=np.array([1.0, 0.0, 0.0])
        )
        
        global_mesh = scene.assemble()
        
        assert global_mesh.num_panels == 4
        assert len(np.unique(global_mesh.component_ids)) == 1
    
    def test_two_component_scene(self):
        # Two squares at different positions
        mesh1 = generate_rectangle(width=1.0, height=1.0)
        mesh2 = generate_rectangle(width=1.0, height=1.0)
        
        transform1 = Transform.from_2d(tx=-2.0, ty=0.0, angle_deg=0.0)
        transform2 = Transform.from_2d(tx=2.0, ty=0.0, angle_deg=0.0)
        
        comp1 = Component("square_left", mesh1, transform1, "wall")
        comp2 = Component("square_right", mesh2, transform2, "wall")
        
        scene = Scene(
            name="two_squares",
            components=[comp1, comp2],
            freestream=np.array([1.0, 0.0, 0.0])
        )
        
        global_mesh = scene.assemble()
        
        assert global_mesh.num_panels == 8  # 4 + 4
        assert global_mesh.num_nodes == 8   # 4 + 4 (no shared nodes)
        
        # Check component IDs
        assert np.sum(global_mesh.component_ids == 0) == 4
        assert np.sum(global_mesh.component_ids == 1) == 4


class TestIO:
    """Test IO functionality."""
    
    def test_read_json(self):
        # Assume square_unit.json exists
        json_path = Path(__file__).parent.parent / "data/geometries/square_unit.json"
        
        if json_path.exists():
            mesh = GeometryReader.read_json(json_path)
            
            assert mesh.is_2d
            assert mesh.num_nodes == 4
            assert mesh.num_panels == 4
    
    def test_case_loader(self):
        # Assume single_square.yaml exists
        case_path = Path(__file__).parent.parent / "cases/single_square.yaml"
        
        if case_path.exists():
            scene, config = CaseLoader.load(case_path)
            
            assert scene.name == "Single Square Test"
            assert scene.num_components == 1
            assert config.solver.type == "constant_source"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
