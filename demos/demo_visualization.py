"""
Demo: Visualization capabilities.

Shows how to plot meshes, components, and scenes.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path (go up to project root, then into src)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.io import generate_rectangle, generate_circle, CaseLoader
from core.geometry import Transform, Component, Scene
from visualization import MeshPlotter, quick_plot_mesh, quick_plot_component, quick_plot_scene


def demo_basic_mesh_plot():
    """Demo 1: Plot a basic mesh."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Mesh Plot")
    print("="*60)
    
    # Create a circle mesh
    mesh = generate_circle(radius=1.0, num_panels=32)
    
    print(f"Created circle mesh with {mesh.num_panels} panels")
    print("Close the plot window to continue...")
    
    # Quick plot (convenience function)
    quick_plot_mesh(mesh, show_normals=True)


def demo_component_transform():
    """Demo 2: Plot component before/after transform."""
    print("\n" + "="*60)
    print("DEMO 2: Component Transformation")
    print("="*60)
    
    # Create a rectangle
    local_mesh = generate_rectangle(width=1.0, height=0.5, center=(0, 0))
    
    # Transform: translate and rotate
    transform = Transform.from_2d(tx=2.0, ty=1.5, angle_deg=30.0)
    
    component = Component(
        name="rotated_rectangle",
        local_mesh=local_mesh,
        transform=transform,
        bc_type="wall"
    )
    
    print(f"Component: {component.name}")
    print(f"  Transform: translation=({transform.translation[0]:.1f}, {transform.translation[1]:.1f}), rotation=30°")
    print("  Blue dashed = local coordinates")
    print("  Red solid = global coordinates")
    print("Close the plot window to continue...")
    
    # Plot showing both local and global
    quick_plot_component(component, show_normals=True)


def demo_multi_component_scene():
    """Demo 3: Plot a scene with multiple components."""
    print("\n" + "="*60)
    print("DEMO 3: Multi-Component Scene")
    print("="*60)
    
    # Create three different shapes
    mesh1 = generate_rectangle(width=1.0, height=1.0)
    mesh2 = generate_circle(radius=0.4, num_panels=24)
    mesh3 = generate_rectangle(width=0.8, height=1.5)
    
    # Place them at different positions and orientations
    comp1 = Component(
        name="square",
        local_mesh=mesh1,
        transform=Transform.from_2d(tx=-2.5, ty=0.0, angle_deg=0.0),
        bc_type="wall"
    )
    
    comp2 = Component(
        name="circle",
        local_mesh=mesh2,
        transform=Transform.from_2d(tx=0.0, ty=0.5, angle_deg=0.0),
        bc_type="wall"
    )
    
    comp3 = Component(
        name="tall_rect",
        local_mesh=mesh3,
        transform=Transform.from_2d(tx=2.5, ty=-0.2, angle_deg=15.0),
        bc_type="wall"
    )
    
    # Create scene
    scene = Scene(
        name="three_shapes_demo",
        components=[comp1, comp2, comp3],
        freestream=np.array([1.5, 0.3, 0.0])
    )
    
    print(f"Scene: {scene.name}")
    print(f"  Components: {scene.num_components}")
    print(f"  Freestream: ({scene.freestream[0]:.2f}, {scene.freestream[1]:.2f})")
    print("  Each component shown in different color")
    print("Close the plot window to continue...")
    
    # Plot entire scene
    quick_plot_scene(scene, show_normals=True)


def demo_advanced_plotting():
    """Demo 4: Advanced plotting with customization."""
    print("\n" + "="*60)
    print("DEMO 4: Advanced Plotting Features")
    print("="*60)
    
    # Create two squares at distance
    mesh1 = generate_rectangle(width=1.0, height=1.0, num_panels_x=2, num_panels_y=2)
    mesh2 = generate_rectangle(width=1.0, height=1.0, num_panels_x=2, num_panels_y=2)
    
    comp1 = Component("left", mesh1, Transform.from_2d(-2.0, 0.0, 0.0), "wall")
    comp2 = Component("right", mesh2, Transform.from_2d(2.0, 0.0, 45.0), "wall")
    
    scene = Scene(
        name="two_squares",
        components=[comp1, comp2],
        freestream=np.array([1.0, 0.0, 0.0])
    )
    
    print(f"Using MeshPlotter class for more control...")
    print("Close the plot window to continue...")
    
    # Use MeshPlotter for more control
    plotter = MeshPlotter(figsize=(12, 8))
    
    # Plot with all features enabled
    plotter.plot_scene(
        scene,
        show_normals=True,
        show_centers=True,
        show_freestream=True
    )
    
    # Optionally save to file
    # plotter.save("output/scene_plot.png")
    
    plotter.show()


def demo_case_file_visualization():
    """Demo 5: Visualize a case from YAML file."""
    print("\n" + "="*60)
    print("DEMO 5: Case File Visualization")
    print("="*60)
    
    case_path = Path(__file__).parent.parent / "cases/two_squares.yaml"
    
    if not case_path.exists():
        print(f"⚠ Case file not found: {case_path}")
        print("Skipping this demo.")
        return
    
    # Load case
    scene, config = CaseLoader.load(case_path)
    
    print(f"Loaded case: {config.name}")
    print(f"  Description: {config.description}")
    print(f"  Components: {scene.num_components}")
    print("Close the plot window to continue...")
    
    # Visualize
    quick_plot_scene(scene, show_normals=True)


def main():
    """Run all visualization demos."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  PANEL METHOD SOLVER - VISUALIZATION DEMO  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    print("\nThis demo shows various visualization capabilities.")
    print("Close each plot window to proceed to the next demo.\n")
    
    try:
        demo_basic_mesh_plot()
        demo_component_transform()
        demo_multi_component_scene()
        demo_advanced_plotting()
        demo_case_file_visualization()
        
        print("\n" + "="*60)
        print("✓ ALL VISUALIZATION DEMOS COMPLETED")
        print("="*60)
        print("\nVisualization features:")
        print("  • quick_plot_mesh(mesh) - Quick mesh visualization")
        print("  • quick_plot_component(component) - Show local & global")
        print("  • quick_plot_scene(scene) - Full scene with components")
        print("  • MeshPlotter class - Advanced control & customization")
        print()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
