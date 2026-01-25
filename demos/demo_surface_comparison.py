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


def extract_panel_surface(case) -> 'SurfaceData':
    """
    Run panel method and extract surface data.
    
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
    
    # Extract surface data
    extractor = SurfaceDataExtractor(case.mesh, solver)
    surface_data = extractor.extract(arc_length=True)
    
    print(f"  Surface points: {len(surface_data.x)}")
    
    return surface_data


def extract_openfoam_surface(
    openfoam_case_dir: Path,
    panel_surface: 'SurfaceData',
    v_inf: float,
    density: float = 1.0
) -> 'SurfaceData':
    """
    Extract OpenFOAM surface data at panel center locations.
    
    Uses interpolation to sample OpenFOAM fields at the same
    points where panel method provides data.
    
    Args:
        openfoam_case_dir: Path to OpenFOAM case
        panel_surface: Panel method surface data (for sampling points)
        v_inf: Freestream velocity
        density: Fluid density
    
    Returns:
        SurfaceData with interpolated OpenFOAM values
    """
    print("\n" + "="*60)
    print("OpenFOAM: Extracting Surface Data")
    print("="*60)
    
    # First, ensure writeCellCentres has been run
    runner = OpenFOAMRunner(openfoam_case_dir, verbose=True)
    print("  Running writeCellCentres...")
    result = runner.run_write_cell_centres()
    if not result.success:
        print(f"  WARNING: writeCellCentres failed: {result.stderr}")
        print("  Attempting to continue anyway...")
    else:
        print("  ✓ writeCellCentres complete")
    
    # Create extractor
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
    
    # Sample at panel center locations
    points = np.column_stack([panel_surface.x, panel_surface.y])
    
    print(f"  Sampling at {len(points)} panel centers...")
    try:
        surface_data = extractor.sample_at_points(
            points,
            reference_pressure=0.0,
            density=density,
            v_inf=v_inf
        )
    except FileNotFoundError as e:
        # Check if U or p fields are missing
        if 'U' in str(e) or 'p' in str(e):
            # Check if this is a parallel case that needs reconstruction
            processor_dirs = list(openfoam_case_dir.glob("processor*"))
            if processor_dirs:
                raise RuntimeError(
                    f"\nMissing flow fields in OpenFOAM case!\n"
                    f"This appears to be a parallel case with {len(processor_dirs)} processor directories.\n"
                    f"The results need to be reconstructed:\n\n"
                    f"  cd {openfoam_case_dir}\n"
                    f"  reconstructPar\n\n"
                    f"Or use a non-parallel OpenFOAM case (one without processor* directories)."
                )
            else:
                raise RuntimeError(
                    f"\nMissing flow fields in OpenFOAM case!\n"
                    f"The case may not have been solved, or fields were cleaned.\n"
                    f"Try running:\n\n"
                    f"  cd {openfoam_case_dir}\n"
                    f"  potentialFoam\n"
                )
        raise RuntimeError(f"Failed to sample OpenFOAM fields: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to sample OpenFOAM fields: {e}")
    
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
    case = CaseLoader.load_case(args.case_dir)
    
    # Extract panel method surface
    panel_surface = extract_panel_surface(case)
    
    # Extract OpenFOAM surface
    openfoam_surface = extract_openfoam_surface(
        args.openfoam_case,
        panel_surface,
        v_inf=case.v_inf,
        density=case.density
    )
    
    # Compare
    print("\n" + "="*60)
    print("Comparison and Visualization")
    print("="*60)
    
    viz = ComparisonVisualizer(output_dir=args.output)
    
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
            interpolate=False  # Already at same points
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
