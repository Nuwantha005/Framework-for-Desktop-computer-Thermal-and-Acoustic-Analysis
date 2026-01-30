#!/usr/bin/env python3
"""
Demo: Test the new foamlib-based case generator.

This script demonstrates how the new template-based generator works
and verifies it creates valid OpenFOAM cases.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.io import CaseLoader
from validation.adapters.openfoam import FoamlibCaseGenerator, MeshSettings

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test foamlib-based OpenFOAM case generator"
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=Path("cases/single_square"),
        help="Path to case directory (default: cases/single_square)"
    )
    parser.add_argument(
        "--mesh-level",
        type=int,
        default=-1,
        help="Mesh level index for parametric cases (default: -1 = finest)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation_results/test_foamlib/openfoam"),
        help="Output directory for OpenFOAM case"
    )
    
    args = parser.parse_args()
    
    if not args.case.exists():
        print(f"ERROR: Case not found: {args.case}")
        print("Please check the path")
        return 1
    
    print("=" * 70)
    print("Testing Foamlib-Based OpenFOAM Case Generator")
    print("=" * 70)
    
    # Load panel method case at specified mesh level
    print(f"\n1. Loading panel method case: {args.case}")
    print(f"   Mesh level: {args.mesh_level} ({'finest' if args.mesh_level == -1 else f'level {args.mesh_level}'})")
    case = CaseLoader.load_case(args.case, mesh_level_index=args.mesh_level)
    print(f"   ✓ Loaded: {case.name}")
    
    # Display mesh info for parametric cases
    if case.num_mesh_levels > 0:
        print(f"   Available mesh levels: {case.num_mesh_levels}")
        print(f"   Current resolution: {case.mesh_level}")
    
    print(f"   Components: {case.mesh.num_panels} panels")
    print(f"   Freestream: {case.freestream}")
    
    # Setup output directory
    print(f"\n2. Creating OpenFOAM case at: {args.output}")
    
    # Create generator with moderate mesh settings
    mesh_settings = MeshSettings(
        background_cells_per_unit=8.0,
        refinement_level=2,
        z_thickness=0.1
    )
    
    generator = FoamlibCaseGenerator(
        case=case,
        output_dir=args.output,
        mesh_settings=mesh_settings,
        n_processors=4  # For parallel snappyHexMesh
    )
    
    # Generate case
    print("\n3. Generating OpenFOAM case files...")
    of_case_dir = generator.generate()
    print(f"   ✓ Generated: {of_case_dir}")
    
    # Verify structure
    print("\n4. Verifying case structure...")
    required_files = [
        "0/U",
        "0/p",
        "constant/transportProperties",
        "constant/triSurface",
        "system/blockMeshDict",
        "system/snappyHexMeshDict",
        "system/controlDict",
        "system/fvSchemes",
        "system/fvSolution",
        "system/decomposeParDict",
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = of_case_dir / file_path
        exists = full_path.exists()
        status = "✓" if exists else "✗"
        print(f"   {status} {file_path}")
        if not exists:
            all_exist = False
    
    # Check STL files
    stl_dir = of_case_dir / "constant" / "triSurface"
    stl_files = list(stl_dir.glob("*.stl"))
    print(f"\n5. Geometry files:")
    for stl in stl_files:
        print(f"   ✓ {stl.name}")
    
    # Summary
    print("\n" + "=" * 70)
    if all_exist and len(stl_files) > 0:
        print("✓ SUCCESS: OpenFOAM case generated successfully!")
        print("\nNext steps:")
        print(f"  cd {of_case_dir}")
        print(f"  blockMesh")
        print(f"  surfaceFeatureExtract")
        print(f"  snappyHexMesh -overwrite  # Or: decomposePar && mpirun snappyHexMesh -parallel && reconstructPar")
        print(f"  checkMesh")
        print(f"  potentialFoam")
        return 0
    else:
        print("✗ FAILED: Some files missing or no geometry generated")
        return 1

if __name__ == "__main__":
    sys.exit(main())
