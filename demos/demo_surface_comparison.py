#!/usr/bin/env python3
"""
Surface Velocity Comparison Demo

Demonstrates tangential velocity comparison between panel method and OpenFOAM.
This is a key validation metric for inviscid flow solvers.

Usage:
    python demo_surface_comparison.py <case_dir> <openfoam_case_dir>

Example:
    python demo_surface_comparison.py ../cases/two_rounded_rects ../cases/two_rounded_rects/out/openfoam_case
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.io import CaseLoader
from solvers.panel2d.spm import SourcePanelSolver
from postprocessing.surface import SurfaceDataExtractor
from visualization.comparison import ComparisonVisualizer
from validation.adapters.openfoam import OpenFOAMSurfaceExtractor, OpenFOAMRunner


def extract_panel_surface(case, openfoam_case_dir: Path) -> 'SurfaceData':
    """
    Run panel method and extract surface data.
    
    Args:
        case: Panel method case
        openfoam_case_dir: OpenFOAM case directory (to find STL reference geometry)
    
    Returns:
        SurfaceData with Vt, Cp at panel centers
    """
    print("\n" + "="*60)
    print("Panel Method: Solving and Extracting Surface Data")
    print("="*60)
    
    # Run solver
    print(f"  Mesh: {case.num_panels} panels, {case.num_components} components")
    solver = SourcePanelSolver(case.mesh, v_inf=case.v_inf, aoa=case.aoa)
    solver.solve()
    
    print(f"  Cp range: [{solver.Cp.min():.4f}, {solver.Cp.max():.4f}]")
    print(f"  Vt range: [{solver.Vt.min():.4f}, {solver.Vt.max():.4f}]")
    
    # Find STL file for geometry projection
    stl_dir = openfoam_case_dir / "constant" / "triSurface"
    stl_files = list(stl_dir.glob("*.stl"))
    reference_stl = str(stl_files[0]) if stl_files else None
    
    # Extract surface data with geometry projection
    extractor = SurfaceDataExtractor(case.mesh, solver)
    surface_data = extractor.extract(arc_length=True, reference_geometry=reference_stl)
    
    print(f"  Surface points: {len(surface_data.x)}")
    if reference_stl:
        print(f"  Using reference geometry: {Path(reference_stl).name}")
    
    return surface_data


def extract_openfoam_surface(
    openfoam_case_dir: Path,
    patch_name: str,
    v_inf: float,
    density: float = 1.0
) -> 'SurfaceData':
    """
    Extract OpenFOAM surface data using proper surfaceFieldValue extraction.
    
    Uses OpenFOAM's surfaceFieldValue function object to get true face-centered
    boundary values (not interpolated cell centers). This is the recommended
    approach for accurate comparison.
    
    Args:
        openfoam_case_dir: Path to OpenFOAM case
        patch_name: Name of wall patch to extract
        v_inf: Freestream velocity
        density: Fluid density
    
    Returns:
        SurfaceData with true wall boundary values
    """
    print("\n" + "="*60)
    print("OpenFOAM: Extracting Surface Data")
    print("="*60)
    
    # Run postProcess to extract surface data
    runner = OpenFOAMRunner(openfoam_case_dir, verbose=True)
    print("  Running postProcess to extract surface data...")
    result = runner.run_post_process(fields=['U', 'p'])
    if not result.success:
        print(f"  WARNING: postProcess failed: {result.stderr}")
        print("  Attempting to continue anyway...")
    else:
        print("  ✓ postProcess complete")
    
    # Create extractor and extract surface data
    try:
        extractor = OpenFOAMSurfaceExtractor(openfoam_case_dir, time_idx=-1)
    except FileNotFoundError as e:
        # Check if this is a parallel case that needs reconstruction
        processor_dirs = list(openfoam_case_dir.glob("processor*"))
        if processor_dirs:
            raise RuntimeError(
                f"OpenFOAM case appears to be parallel (found {len(processor_dirs)} processor directories).\n"
                f"The fields may need to be reconstructed. Try:\n"
                f"  cd {openfoam_case_dir}\n"
                f"  reconstructPar\n"
                f"Or use a non-parallel case."
            )
        raise RuntimeError(f"Failed to create OpenFOAM extractor: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to create OpenFOAM extractor: {e}")
    
    # Extract surface data
    print(f"  Extracting patch: {patch_name}...")
    try:
        surface_data = extractor.extract(
            patch_name=patch_name,
            reference_pressure=0.0,
            density=density,
            v_inf=v_inf
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"\nFailed to extract surface data: {e}\n"
            f"This usually means postProcess needs to be run first.\n"
            f"The case should have function objects configured for surface extraction."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to extract surface data: {e}")
    
    print(f"  Surface points: {len(surface_data.x)}")
    print(f"  Cp range: [{surface_data.Cp.min():.4f}, {surface_data.Cp.max():.4f}]")
    print(f"  Vt range: [{surface_data.Vt.min():.4f}, {surface_data.Vt.max():.4f}]")
    
    return surface_data


def main():
    """Main workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare panel method and OpenFOAM surface distributions"
    )
    parser.add_argument(
        "case_dir",
        type=Path,
        help="Path to panel method case directory"
    )
    parser.add_argument(
        "openfoam_case",
        type=Path,
        help="Path to OpenFOAM case directory"
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
        default=None,
        help="Output directory for plots (default: case_dir/out)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.case_dir.exists():
        print(f"Error: Case directory not found: {args.case_dir}")
        sys.exit(1)
    
    if not args.openfoam_case.exists():
        print(f"Error: OpenFOAM case not found: {args.openfoam_case}")
        sys.exit(1)
    
    # Set output directory
    if args.output is None:
        args.output = args.case_dir / "out"
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Load case
    print(f"\nLoading case: {args.case_dir}")
    print(f"  Using mesh level: {args.mesh_level} ({'finest' if args.mesh_level == -1 else f'level {args.mesh_level}'})")
    case = CaseLoader.load_case(args.case_dir, mesh_level_index=args.mesh_level)
    
    # Display mesh info for parametric cases
    if case.num_mesh_levels > 0:
        print(f"  Available mesh levels: {case.num_mesh_levels}")
        print(f"  Current resolution: {case.mesh_level}")
        print(f"  Total panels: {case.num_panels}")
    
    # Extract panel method surface
    panel_surface = extract_panel_surface(case, args.openfoam_case)
    
    # Determine patch name from case (use sanitized component name)
    from validation.adapters.openfoam.foamlib_generator import sanitize_name
    patch_name = sanitize_name(case.name)
    
    # Extract OpenFOAM surface
    openfoam_surface = extract_openfoam_surface(
        args.openfoam_case,
        patch_name,
        v_inf=case.v_inf,
        density=case.density
    )
    
    # Compare
    print("\n" + "="*60)
    print("Comparison and Visualization")
    print("="*60)
    
    viz = ComparisonVisualizer(output_dir=args.output)
    print("panel usrface s range:", panel_surface.s.min(), panel_surface.s.max())
    print("openfoam surface s range:", openfoam_surface.s.min(), openfoam_surface.s.max())
    # Plot surface distributions
    fig = viz.compare_surface_distributions(
        surface_data_list=[panel_surface, openfoam_surface],
        labels=["Panel Method", "OpenFOAM (potentialFoam)"],
        title="Surface Distribution Comparison",
        quantities=['Vt', 'Cp'],
        show_by_component=False
    )
    
    output_file = args.output / "surface_comparison.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {output_file}")
    
    # Compute metrics
    print("\n" + "="*60)
    print("Error Metrics")
    print("="*60)
    
    for quantity in ['Vt', 'Cp']:
        metrics = viz.compute_surface_metrics(
            panel_surface,
            openfoam_surface,
            quantity=quantity,
            interpolate=True  # Need to interpolate - different sampling points
        )
        
        print(f"\n{quantity}:")
        print(f"  L2 norm:   {metrics['L2']:.6g}")
        print(f"  L∞ norm:   {metrics['Linf']:.6g}")
        print(f"  RMS:       {metrics['RMS']:.6g}")
        print(f"  MAE:       {metrics['MAE']:.6g}")
        print(f"  Relative L2:   {metrics['rel_L2']*100:.2f}%")
        print(f"  Relative L∞:   {metrics['rel_Linf']*100:.2f}%")
    
    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    
    print("\n" + "="*60)
    print("Surface comparison complete!")
    print("="*60)


if __name__ == "__main__":
    main()
