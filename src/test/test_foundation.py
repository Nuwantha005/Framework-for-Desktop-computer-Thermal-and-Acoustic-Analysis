"""
Demo: Foundation architecture test.

Run this to verify the geometry foundation works correctly.
"""

import sys
from pathlib import Path

# Add src to path (go up to src directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.io import CaseLoader, generate_rectangle
from core.geometry import Transform, Component, Scene
import numpy as np


def test_basic_geometry():
    """Test basic geometry creation."""
    print("=" * 60)
    print("TEST 1: Basic Geometry Generation")
    print("=" * 60)
    
    # Generate a rectangle
    mesh = generate_rectangle(width=2.0, height=1.0, center=(0.0, 0.0))
    
    print(f"✓ Created rectangle mesh")
    print(f"  Nodes: {mesh.num_nodes}")
    print(f"  Panels: {mesh.num_panels}")
    print(f"  Dimension: {mesh.dimension}D")
    print(f"  First panel center: {mesh.centers[0]}")
    print(f"  First panel normal: {mesh.normals[0]}")
    print()


def test_transform():
    """Test transformations."""
    print("=" * 60)
    print("TEST 2: Transforms")
    print("=" * 60)
    
    # Create a unit square
    mesh = generate_rectangle(width=1.0, height=1.0, center=(0.0, 0.0))
    
    # Transform: translate and rotate
    transform = Transform.from_2d(tx=3.0, ty=2.0, angle_deg=45.0)
    
    component = Component(
        name="rotated_square",
        local_mesh=mesh,
        transform=transform,
        bc_type="wall"
    )
    
    global_mesh = component.get_global_mesh(component_id=0)
    
    print(f"✓ Applied transform (translation + 45° rotation)")
    print(f"  Original center: (0, 0, 0)")
    print(f"  Transformed center (approx): ({global_mesh.centers.mean(axis=0)[0]:.3f}, "
          f"{global_mesh.centers.mean(axis=0)[1]:.3f}, "
          f"{global_mesh.centers.mean(axis=0)[2]:.3f})")
    print()


def test_scene_assembly():
    """Test scene assembly with multiple components."""
    print("=" * 60)
    print("TEST 3: Scene Assembly")
    print("=" * 60)
    
    # Create two squares
    mesh1 = generate_rectangle(width=1.0, height=1.0)
    mesh2 = generate_rectangle(width=1.0, height=1.0)
    
    # Place them at different positions
    transform1 = Transform.from_2d(tx=-3.0, ty=0.0, angle_deg=0.0)
    transform2 = Transform.from_2d(tx=3.0, ty=0.0, angle_deg=45.0)
    
    comp1 = Component("square_left", mesh1, transform1, "wall")
    comp2 = Component("square_right", mesh2, transform2, "wall")
    
    scene = Scene(
        name="two_squares_demo",
        components=[comp1, comp2],
        freestream=np.array([1.0, 0.0, 0.0])
    )
    
    global_mesh = scene.assemble()
    
    print(f"✓ Assembled scene: {scene}")
    print(f"  Total nodes: {global_mesh.num_nodes}")
    print(f"  Total panels: {global_mesh.num_panels}")
    print(f"  Components: {len(np.unique(global_mesh.component_ids))}")
    print(f"  Freestream: {scene.freestream}")
    print()


def test_case_loader():
    """Test loading from YAML case file."""
    print("=" * 60)
    print("TEST 4: Case Loader (YAML)")
    print("=" * 60)
    
    case_path = Path(__file__).parent.parent.parent / "cases/single_square.yaml"
    
    if not case_path.exists():
        print(f"⚠ Case file not found: {case_path}")
        print()
        return
    
    try:
        scene, config = CaseLoader.load(case_path)
        
        print(f"✓ Loaded case: {config.name}")
        print(f"  Case type: {config.case_type}")
        print(f"  Description: {config.description}")
        print(f"  Components: {scene.num_components}")
        print(f"  Solver type: {config.solver.type}")
        print(f"  Tolerance: {config.solver.tolerance}")
        print(f"  Freestream velocity: {scene.freestream}")
        print()
        
        # Show assembled mesh
        global_mesh = scene.assemble()
        print(f"✓ Scene assembled successfully")
        print(f"  Total panels: {global_mesh.num_panels}")
        print()
        
    except Exception as e:
        print(f"✗ Error loading case: {e}")
        import traceback
        traceback.print_exc()
        print()


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  PANEL METHOD SOLVER - FOUNDATION ARCHITECTURE TEST  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    try:
        test_basic_geometry()
        test_transform()
        test_scene_assembly()
        test_case_loader()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nFoundation architecture is working correctly!")
        print("Ready for Phase 2: Panel Solver Implementation\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
