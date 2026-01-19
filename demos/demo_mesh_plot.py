#!/usr/bin/env python3
"""
Demo: Mesh Plotting

Loads a case file and visualizes the mesh geometry.
Usage:
    python demo_mesh_plot.py <case_dir> [--show] [--save] [--normals]
    
Example:
    python demo_mesh_plot.py ../cases/cylinder_flow --show
    python demo_mesh_plot.py ../cases/single_square --save --normals
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.io import CaseLoader
from visualization import Visualizer


def main():
    parser = argparse.ArgumentParser(description="Plot mesh from case file")
    parser.add_argument("case_dir", type=str, help="Path to case directory (contains case.yaml)")
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    parser.add_argument("--save", action="store_true", help="Save plot to case_dir/out/")
    parser.add_argument("--normals", action="store_true", help="Show panel normals")
    parser.add_argument("--protect", action="store_true", help="Save to timestamped subfolder")
    args = parser.parse_args()
    
    # Default: save if neither specified
    if not args.show and not args.save:
        args.save = True
    
    # Load case using the cleaner API
    case_dir = Path(args.case_dir).resolve()
    case = CaseLoader.load_case(case_dir)
    
    print(f"Loaded: {case.name}")
    print(f"  Components: {case.num_components}")
    print(f"  Panels: {case.num_panels}")
    
    # Setup visualizer
    output_dir = case.output_dir if args.save else None
    viz = Visualizer(output_dir=output_dir, protect_overwrite=args.protect)
    
    # Plot
    show_normals = args.normals or case.show_normals
    viz.create_figure(figsize=(10, 8))
    viz.plot_scene(case.scene, show_normals=show_normals, show_freestream=True,
                   title=f"{case.name} - Mesh")
    
    save_name = "mesh.png" if args.save else None
    viz.finalize(save=save_name, show=args.show)
    
    print("Done.")


if __name__ == "__main__":
    main()
