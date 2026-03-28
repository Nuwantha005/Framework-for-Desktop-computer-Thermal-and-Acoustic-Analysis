#!/usr/bin/env python3
"""
Compare panel method vs OpenFOAM surface distributions.

Extracts surface data (tangential velocity Vt, pressure coefficient Cp) from both
solvers and generates comparison visualizations using existing extractors and 
ComparisonVisualizer.

Uses existing modules:
    - postprocessing.surface.SurfaceDataExtractor (panel method)
    - validation.adapters.openfoam.OpenFOAMSurfaceExtractor (OpenFOAM)
    - visualization.comparison.ComparisonVisualizer (plotting + metrics)

Usage:
    python validation/scripts/compare_surface.py cases/case_name \\
        --of-case of_case/cases/level_2 --mesh-level -1
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validation.scripts.utils import (
    load_viz_config,
    save_surface_data,
    save_error_metrics,
)


def extract_panel_surface(case, openfoam_case_dir: Path):
    """
    Run panel method solver and extract surface data.
    
    Args:
        case: Panel method Case object
        openfoam_case_dir: OpenFOAM case directory (to find STL reference geometry)
    
    Returns:
        SurfaceData with Vt, Cp at panel centers
    """
    from solvers.panel2d import LinearVortexPanelSolver
    from postprocessing.surface import SurfaceDataExtractor
    
    print(f"  Mesh: {case.num_panels} panels, {case.num_components} components")
    
    # Run solver
    solver = LinearVortexPanelSolver(case.mesh, v_inf=case.v_inf, aoa=case.aoa)
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
    patch_names: List[str],
    v_inf: float,
    density: float = 1.0
):
    """
    Extract OpenFOAM surface data for one or more patches.
    
    Args:
        openfoam_case_dir: Path to OpenFOAM case
        patch_names: List of wall patch names to extract
        v_inf: Freestream velocity
        density: Fluid density
    
    Returns:
        SurfaceData with data from all patches (component_id identifies each)
    """
    from validation.adapters.openfoam import OpenFOAMSurfaceExtractor, OpenFOAMRunner
    from postprocessing.surface import SurfaceData
    
    # Run postProcess to extract surface data
    runner = OpenFOAMRunner(openfoam_case_dir, verbose=True)
    print("  Running postProcess to extract surface data...")
    result = runner.run_post_process(fields=['U', 'p'])
    if not result.success:
        print(f"  WARNING: postProcess failed: {result.stderr}")
        print("  Attempting to continue anyway...")
    else:
        print("  ✓ postProcess complete")
    
    # Create extractor
    try:
        extractor = OpenFOAMSurfaceExtractor(openfoam_case_dir, time_idx=-1)
    except FileNotFoundError as e:
        processor_dirs = list(openfoam_case_dir.glob("processor*"))
        if processor_dirs:
            raise RuntimeError(
                f"OpenFOAM case appears to be parallel (found {len(processor_dirs)} processor directories).\n"
                f"Try: cd {openfoam_case_dir} && reconstructPar"
            )
        raise RuntimeError(f"Failed to create OpenFOAM extractor: {e}")
    
    # Extract each patch
    surface_data_list = []
    for comp_id, patch_name in enumerate(patch_names):
        print(f"  Extracting patch: {patch_name}...")
        try:
            data = extractor.extract(
                patch_name=patch_name,
                reference_pressure=0.0,
                density=density,
                v_inf=v_inf
            )
            # Set component ID for this patch
            data.component_id = np.full(len(data.x), comp_id, dtype=np.int32)
            surface_data_list.append(data)
            
            print(f"    Points: {len(data.x)}, Cp: [{data.Cp.min():.4f}, {data.Cp.max():.4f}]")
        except FileNotFoundError as e:
            print(f"  WARNING: Could not extract patch {patch_name}: {e}")
            continue
        except Exception as e:
            print(f"  WARNING: Error extracting patch {patch_name}: {e}")
            continue
    
    if not surface_data_list:
        raise RuntimeError("No surface data extracted from OpenFOAM")
    
    # Concatenate all patches
    if len(surface_data_list) == 1:
        return surface_data_list[0]
    
    # Multi-component concatenation
    x = np.concatenate([d.x for d in surface_data_list])
    y = np.concatenate([d.y for d in surface_data_list])
    s = np.concatenate([d.s for d in surface_data_list])
    Vt = np.concatenate([d.Vt for d in surface_data_list])
    Cp = np.concatenate([d.Cp for d in surface_data_list])
    Vn = np.concatenate([d.Vn for d in surface_data_list]) if surface_data_list[0].Vn is not None else None
    component_id = np.concatenate([d.component_id for d in surface_data_list])
    
    return SurfaceData(
        x=x, y=y, s=s, Vt=Vt, Vn=Vn, Cp=Cp,
        component_id=component_id,
        source="openfoam"
    )


def main():
    parser = argparse.ArgumentParser(description="Compare panel vs OpenFOAM surface distributions")
    parser.add_argument("case_dir", type=Path, help="Path to case directory")
    parser.add_argument("--of-case", type=Path, required=True, 
                       help="Path to OpenFOAM case (e.g., of_case/cases/level_2)")
    parser.add_argument("--mesh-level", type=int, default=-1,
                       help="Panel mesh level index (-1 for finest)")
    parser.add_argument("--quantities", nargs='+', default=['Vt', 'Cp'],
                       help="Quantities to compare")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--by-component", action="store_true", 
                       help="Plot each component separately")
    
    args = parser.parse_args()
    
    case_dir = args.case_dir.resolve()
    of_case_dir = (case_dir / args.of_case).resolve() if not args.of_case.is_absolute() else args.of_case
    
    if not case_dir.exists():
        print(f"Error: Case directory not found: {case_dir}")
        return 1
    
    if not of_case_dir.exists():
        print(f"Error: OpenFOAM case not found: {of_case_dir}")
        return 1
    
    # Determine OF level name for output organization
    of_level = of_case_dir.name.replace("level_", "")
    
    # Load viz config if exists
    viz_config_path = case_dir / "out" / "viz_config.yaml"
    viz_config = load_viz_config(viz_config_path) if viz_config_path.exists() else {}
    
    # Load panel case
    from core.io import CaseLoader
    print(f"\nLoading case: {case_dir}")
    print(f"  Using mesh level: {args.mesh_level} ({'finest' if args.mesh_level == -1 else f'level {args.mesh_level}'})")
    
    case = CaseLoader.load_case(case_dir, mesh_level_index=args.mesh_level)
    
    # Display mesh info for parametric cases
    if case.num_mesh_levels > 0:
        print(f"  Available mesh levels: {case.num_mesh_levels}")
        print(f"  Current resolution: {case.mesh_level}")
        print(f"  Total panels: {case.num_panels}")
    
    v_inf = case.v_inf
    rho = case.config.fluid.density
    
    print(f"\nRunning surface comparison...")
    print(f"  Panel case: {case_dir}")
    print(f"  OpenFOAM case: {of_case_dir}")
    
    # Run panel solver and extract surface data
    print(f"\n{'='*60}")
    print("Panel Method: Solving and Extracting Surface Data")
    print("="*60)
    panel_surface = extract_panel_surface(case, of_case_dir)
    
    # Determine patch names from components
    # For single component cases, the patch name is the sanitized case name
    # For multi-component cases, each component has its own patch
    from validation.adapters.openfoam.foamlib_generator import sanitize_name
    
    if case.num_components == 1:
        # Single component - patch name is the case name
        patch_names = [sanitize_name(case.name)]
    else:
        # Multi-component - each component is a separate patch
        patch_names = [sanitize_name(comp.name) for comp in case.scene.components]
    
    print(f"\nPatch names for extraction: {patch_names}")
    
    # Extract OpenFOAM surface data
    print(f"\n{'='*60}")
    print("OpenFOAM: Extracting Surface Data")
    print("="*60)
    openfoam_surface = extract_openfoam_surface(
        of_case_dir,
        patch_names,
        v_inf=v_inf,
        density=rho
    )
    
    # Save raw data
    output_dir = case_dir / "out"
    print(f"\nSaving raw data...")
    save_surface_data(output_dir, 
                     {'all': {'s': panel_surface.s, 'Vt': panel_surface.Vt, 
                              'Cp': panel_surface.Cp, 'x': panel_surface.x, 'y': panel_surface.y}},
                     "panel", of_level=of_level)
    save_surface_data(output_dir,
                     {'all': {'s': openfoam_surface.s, 'Vt': openfoam_surface.Vt,
                              'Cp': openfoam_surface.Cp, 'x': openfoam_surface.x, 'y': openfoam_surface.y}},
                     "openfoam", of_level=of_level)
    
    # Compare and visualize using ComparisonVisualizer
    print(f"\n{'='*60}")
    print("Comparison and Visualization")
    print("="*60)
    
    from visualization.comparison import ComparisonVisualizer
    
    plots_dir = output_dir / "surface_comparison" / f"of_{of_level}" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    viz = ComparisonVisualizer(output_dir=plots_dir)
    
    print(f"  Panel surface s range: [{panel_surface.s.min():.4f}, {panel_surface.s.max():.4f}]")
    print(f"  OpenFOAM surface s range: [{openfoam_surface.s.min():.4f}, {openfoam_surface.s.max():.4f}]")
    
    # Plot surface distributions
    fig = viz.compare_surface_distributions(
        surface_data_list=[panel_surface, openfoam_surface],
        labels=["Panel Method", "OpenFOAM (potentialFoam)"],
        title="Surface Distribution Comparison",
        quantities=args.quantities,
        show_by_component=args.by_component
    )
    
    output_file = plots_dir / "surface_comparison.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {output_file}")
    
    # Compute error metrics
    print(f"\n{'='*60}")
    print("Error Metrics")
    print("="*60)
    
    error_metrics = {}
    for quantity in args.quantities:
        metrics = viz.compute_surface_metrics(
            panel_surface,
            openfoam_surface,
            quantity=quantity,
            interpolate=True  # Need to interpolate - different sampling points
        )
        
        error_metrics[quantity] = {
            'L2': float(metrics['L2']),
            'Linf': float(metrics['Linf']),
            'RMS': float(metrics['RMS']),
            'MAE': float(metrics['MAE']),
            'relative_L2': float(metrics['rel_L2']),
            'relative_Linf': float(metrics['rel_Linf']),
        }
        
        print(f"\n{quantity}:")
        print(f"  L2 norm:   {metrics['L2']:.6g}")
        print(f"  L∞ norm:   {metrics['Linf']:.6g}")
        print(f"  RMS:       {metrics['RMS']:.6g}")
        print(f"  MAE:       {metrics['MAE']:.6g}")
        print(f"  Relative L2:   {metrics['rel_L2']*100:.2f}%")
        print(f"  Relative L∞:   {metrics['rel_Linf']*100:.2f}%")
    
    save_error_metrics(output_dir, {'surface': error_metrics}, "surface_comparison", of_level=of_level)
    
    # Generate surface envelope plots
    print(f"\n{'='*60}")
    print("Generating Surface Envelope Plots")
    print("="*60)
    
    from visualization.surface_envelope import plot_surface_envelope, plot_dual_surface_envelope
    from scipy.interpolate import interp1d
    import matplotlib.pyplot as plt
    
    envelope_scale = 0.3
    
    for quantity in args.quantities:
        panel_vals = getattr(panel_surface, quantity)
        of_vals = getattr(openfoam_surface, quantity)
        
        # Determine if we should invert (for Cp)
        invert = (quantity == 'Cp')
        cmap = 'viridis' if quantity == 'Vt' else 'RdBu_r'
        
        # 1. Panel method envelope
        fig, ax = plot_surface_envelope(
            panel_surface.x, panel_surface.y, panel_vals,
            scale=envelope_scale,
            quantity_name=quantity,
            colormap=cmap,
            invert_values=invert,
            title=f'Panel Method - {quantity} Surface Distribution',
            whisker_density=max(1, len(panel_surface.x) // 40),
        )
        fig.savefig(plots_dir / f"envelope_{quantity}_panel.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {quantity} panel envelope")
        
        # 2. OpenFOAM envelope
        fig, ax = plot_surface_envelope(
            openfoam_surface.x, openfoam_surface.y, of_vals,
            scale=envelope_scale,
            quantity_name=quantity,
            colormap=cmap,
            invert_values=invert,
            title=f'OpenFOAM - {quantity} Surface Distribution',
            whisker_density=max(1, len(openfoam_surface.x) // 40),
        )
        fig.savefig(plots_dir / f"envelope_{quantity}_openfoam.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {quantity} OpenFOAM envelope")
        
        # 3. Comparison envelope (interpolate OF to panel mesh)
        of_interp = interp1d(openfoam_surface.s, of_vals, kind='linear',
                           bounds_error=False, fill_value='extrapolate')
        of_vals_interp = of_interp(panel_surface.s)
        
        fig, ax = plot_dual_surface_envelope(
            panel_surface.x, panel_surface.y,
            panel_vals, of_vals_interp,
            label1='Panel Method',
            label2='OpenFOAM',
            scale=envelope_scale,
            quantity_name=quantity,
            invert_values=invert,
            title=f'{quantity} Distribution Comparison (Envelope)',
        )
        fig.savefig(plots_dir / f"envelope_{quantity}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {quantity} comparison envelope")
    
    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    
    print(f"\n{'='*60}")
    print("Surface comparison complete!")
    print(f"Results saved to: {output_dir / 'surface_comparison' / f'of_{of_level}'}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
