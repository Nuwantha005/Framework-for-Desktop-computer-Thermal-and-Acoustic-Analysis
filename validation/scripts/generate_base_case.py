#!/usr/bin/env python3
"""
Step 1: Generate Base OpenFOAM Case

Generates an OpenFOAM case from the panel method case.yaml and saves it
in cases/case_name/of_case/base_case/.

Also creates a default of_case/config.yaml if it doesn't exist.

Usage:
    python generate_base_case.py <case_dir> [options]

Example:
    python generate_base_case.py cases/single_square --mesh-level -1
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.io import CaseLoader
from validation.adapters.openfoam import FoamlibCaseGenerator, MeshSettings
from validation.scripts.utils import create_default_of_config
import yaml


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate base OpenFOAM case for validation"
    )
    parser.add_argument(
        "case_dir",
        type=Path,
        help="Path to panel method case directory"
    )
    parser.add_argument(
        "--mesh-level",
        type=int,
        default=-1,
        help="Panel mesh level to use for geometry (default: -1 = finest)"
    )
    parser.add_argument(
        "--background-cells",
        type=float,
        default=8.0,
        help="Background mesh cells per unit length (default: 8.0)"
    )
    parser.add_argument(
        "--refinement-level",
        type=int,
        default=2,
        help="SnappyHexMesh surface refinement level (default: 2)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing base case"
    )
    
    args = parser.parse_args()
    
    # Validate case directory
    if not args.case_dir.exists():
        print(f"ERROR: Case directory not found: {args.case_dir}")
        return 1
    
    case_yaml = args.case_dir / "case.yaml"
    if not case_yaml.exists():
        print(f"ERROR: case.yaml not found in {args.case_dir}")
        return 1
    
    print("=" * 70)
    print("Step 1: Generate Base OpenFOAM Case")
    print("=" * 70)
    
    # Setup paths
    of_case_dir = args.case_dir / "of_case"
    base_case_dir = of_case_dir / "base_case"
    config_file = of_case_dir / "config.yaml"
    
    # Check if base case exists
    if base_case_dir.exists() and not args.overwrite:
        print(f"\nBase case already exists: {base_case_dir}")
        print("Use --overwrite to regenerate")
        return 1
    
    # Load panel method case
    print(f"\n1. Loading panel method case: {args.case_dir.name}")
    case = CaseLoader.load_case(args.case_dir, mesh_level_index=args.mesh_level)
    print(f"   ✓ Case: {case.name}")
    print(f"   ✓ Components: {case.num_components}")
    print(f"   ✓ Panels: {case.num_panels}")
    
    if case.num_mesh_levels > 0:
        print(f"   ✓ Mesh levels available: {case.num_mesh_levels}")
        print(f"   ✓ Using level: {args.mesh_level} ({case.mesh_level})")
    
    # Generate OpenFOAM case
    print(f"\n2. Generating OpenFOAM case...")
    print(f"   Output: {base_case_dir}")
    
    mesh_settings = MeshSettings(
        background_cells_per_unit=args.background_cells,
        refinement_level=args.refinement_level,
        z_thickness=0.1
    )
    
    generator = FoamlibCaseGenerator(
        case=case,
        output_dir=base_case_dir,
        mesh_settings=mesh_settings,
        n_processors=4
    )
    
    of_case_path = generator.generate()
    print(f"   ✓ Generated: {of_case_path}")
    
    # Create default config if doesn't exist
    if not config_file.exists():
        print(f"\n3. Creating default OpenFOAM convergence config...")
        
        # Get component names for per-component refinement
        component_names = [comp.name for comp in case.scene.components]
        
        config = create_default_of_config(
            case_name=case.name,
            num_levels=4,
            component_names=component_names
        )
        
        of_case_dir.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"   ✓ Created: {config_file}")
        print(f"   ℹ Edit this file to customize convergence study parameters")
    else:
        print(f"\n3. Using existing config: {config_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("✓ Base case generation complete!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  • Base OpenFOAM case: {base_case_dir}")
    print(f"  • Convergence config: {config_file}")
    print(f"\nNext steps:")
    print(f"  1. Review/edit: {config_file}")
    print(f"  2. Run: python validation/scripts/run_of_convergence.py {args.case_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
