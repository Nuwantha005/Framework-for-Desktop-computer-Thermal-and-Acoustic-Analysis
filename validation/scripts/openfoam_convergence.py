#!/usr/bin/env python3
"""
OpenFOAM Mesh Convergence Study

Runs a grid independence test for OpenFOAM (potentialFoam) to determine
the mesh-converged solution that can be used as benchmark for panel method.

The study:
1. Reads refinement levels from a YAML configuration file
2. Creates OpenFOAM cases at each refinement level
3. Runs blockMesh, snappyHexMesh, and potentialFoam for each
4. Extracts velocity field and computes convergence metrics
5. Calculates Grid Convergence Index (GCI) using Richardson extrapolation
6. Saves the final (finest) case for use in panel method comparison

Usage:
    python openfoam_convergence.py <case_path>
    python openfoam_convergence.py cases/two_rounded_rects --show
    
The script expects:
    - An existing OpenFOAM case at validation_results/<case_name>/openfoam/
    - A convergence config at validation_results/<case_name>/openfoam_convergence/config.yaml
      (auto-generated if not present)
"""

import sys
import argparse
import shutil
import subprocess
import re
import time
from pathlib import Path
from typing import Optional
import yaml
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from foamlib import FoamCase, FoamFile
    FOAMLIB_AVAILABLE = True
except ImportError:
    FOAMLIB_AVAILABLE = False
    print("Warning: foamlib not installed. Install with: pip install foamlib")

from scipy.interpolate import griddata
import matplotlib.pyplot as plt


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_CONFIG = """# OpenFOAM Mesh Convergence Study Configuration
# =============================================
# This file controls the grid independence test parameters.
# Modify refinement levels and settings as needed, then re-run the study.

# Refinement Levels
# -----------------
# Each level specifies blockMesh background cells and snappyHexMesh refinement.
# blockMesh provides the background mesh (keep relatively coarse).
# snappyHexMesh refines near surfaces (this drives the accuracy).
#
# For 2D cases: z-cells should stay at 1

refinement_levels:
  - name: "coarse"
    blockMesh_cells: [170, 120, 1]    # [x, y, z] background cells
    snappy_surface_level: 2           # Surface refinement level
    snappy_feature_level: 2           # Feature edge refinement level

  - name: "medium" 
    blockMesh_cells: [200, 150, 1]
    snappy_surface_level: 3
    snappy_feature_level: 3

  - name: "fine"
    blockMesh_cells: [250, 180, 1]
    snappy_surface_level: 4
    snappy_feature_level: 4

  - name: "very_fine"
    blockMesh_cells: [320, 240, 1]
    snappy_surface_level: 5
    snappy_feature_level: 5

# Convergence Settings
# --------------------
convergence:
  gci_threshold: 0.05           # Grid Convergence Index threshold (5%)
  refinement_ratio: 1.5         # Approximate refinement ratio between levels
  
# Comparison Grid (for velocity field comparison)
# -----------------------------------------------
# Structured grid in far-field for extracting convergence metrics
comparison_grid:
  x_range: [-6.0, 6.0]
  y_range: [-4.0, 4.0]
  nx: 80
  ny: 60
  body_distance: 0.5            # Exclude points within this distance of bodies

# Output Settings
# ---------------
output:
  delete_intermediate_cases: true   # Delete intermediate OF cases after metrics extraction
  save_final_case: true             # Copy final (finest) case to openfoam/final_case/
  verbose: true                     # Print detailed progress
"""


def create_default_config(config_path: Path) -> None:
    """Create default configuration file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        f.write(DEFAULT_CONFIG)
    print(f"Created default config: {config_path}")


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# =============================================================================
# OpenFOAM Case Modification
# =============================================================================

def modify_blockmesh_dict(case_path: Path, cells: list[int]) -> None:
    """
    Modify blockMeshDict to use specified cell counts.
    
    Args:
        case_path: Path to OpenFOAM case
        cells: [nx, ny, nz] cell counts
    """
    bm_path = case_path / "system" / "blockMeshDict"
    
    with open(bm_path, 'r') as f:
        content = f.read()
    
    # Replace the hex block cell specification
    # Pattern: hex (vertices) (nx ny nz) simpleGrading
    pattern = r'(hex\s*\([^)]+\)\s*)\(\s*\d+\s+\d+\s+\d+\s*\)'
    replacement = rf'\1({cells[0]} {cells[1]} {cells[2]})'
    new_content = re.sub(pattern, replacement, content)
    
    with open(bm_path, 'w') as f:
        f.write(new_content)


def modify_snappy_dict(case_path: Path, surface_level: int, feature_level: int) -> None:
    """
    Modify snappyHexMeshDict refinement levels.
    
    Args:
        case_path: Path to OpenFOAM case
        surface_level: Surface refinement level
        feature_level: Feature edge refinement level
    """
    snappy_path = case_path / "system" / "snappyHexMeshDict"
    
    with open(snappy_path, 'r') as f:
        content = f.read()
    
    # Update feature refinement levels
    # Pattern: level 2; -> level N;
    content = re.sub(
        r'(file\s+"[^"]+\.eMesh";\s*level\s+)\d+',
        rf'\g<1>{feature_level}',
        content
    )
    
    # Update surface refinement levels
    # Pattern: level (2 2); -> level (N N);
    content = re.sub(
        r'(level\s*)\(\s*\d+\s+\d+\s*\)',
        rf'\1({surface_level} {surface_level})',
        content
    )
    
    # Update maxLocalCells and maxGlobalCells based on refinement level
    # Higher refinement needs more cells allowed
    scale_factor = 2 ** (surface_level - 2)  # Base level is 2
    
    content = re.sub(
        r'(maxLocalCells\s+)\d+',
        rf'\g<1>{100000 * scale_factor}',
        content
    )
    content = re.sub(
        r'(maxGlobalCells\s+)\d+',
        rf'\g<1>{2000000 * scale_factor}',
        content
    )
    
    with open(snappy_path, 'w') as f:
        f.write(content)


# =============================================================================
# OpenFOAM Execution
# =============================================================================

def run_openfoam_command(cmd: list[str], case_path: Path, verbose: bool = True) -> bool:
    """
    Run an OpenFOAM command and optionally display output.
    
    Returns True if successful, False otherwise.
    """
    cmd_str = ' '.join(cmd)
    if verbose:
        print(f"  Running: {cmd_str}")
    
    try:
        if verbose:
            # Stream output in real-time
            process = subprocess.Popen(
                cmd,
                cwd=case_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Print output lines as they come
            residual_pattern = re.compile(r'Time = |Solving for|GAMG:|smoothSolver:|Final residual|ExecutionTime')
            for line in process.stdout:
                line = line.rstrip()
                # Filter to show only important lines
                if residual_pattern.search(line) or 'Error' in line or 'Warning' in line:
                    print(f"    {line}")
            
            process.wait()
            return process.returncode == 0
        else:
            result = subprocess.run(
                cmd,
                cwd=case_path,
                capture_output=True,
                text=True
            )
            return result.returncode == 0
            
    except Exception as e:
        print(f"  Error running {cmd[0]}: {e}")
        return False


def reset_initial_conditions(case_path: Path, v_inf: float = 10.0) -> None:
    """
    Reset initial conditions to uniform values after snappyHexMesh.
    
    snappyHexMesh creates a new mesh with different cell count, so the
    original nonuniform initial conditions become invalid. This function
    rewrites 0/U and 0/p with uniform values that work with any mesh size.
    
    Args:
        case_path: Path to OpenFOAM case
        v_inf: Freestream velocity magnitude (m/s)
    """
    # Get the boundary patches from the new mesh
    # We need to include any new patches created by snappyHexMesh (walls)
    
    # Check what wall patches exist
    boundary_path = case_path / "constant" / "polyMesh" / "boundary"
    wall_patches = []
    
    if boundary_path.exists():
        with open(boundary_path, 'r') as f:
            content = f.read()
            # Find wall type patches (rect_front, rect_back, etc.)
            import re
            # Look for patches followed by type wall
            for match in re.finditer(r'(\w+)\s*\{[^}]*type\s+wall', content):
                wall_patches.append(match.group(1))
    
    # Write U file with uniform initial condition
    # Note: front/back are symmetry type in blockMeshDict (for 2D simulations)
    u_content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform ({v_inf} 0 0);

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform ({v_inf} 0 0);
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    top
    {{
        type            slip;
    }}
    bottom
    {{
        type            slip;
    }}
    front
    {{
        type            symmetry;
    }}
    back
    {{
        type            symmetry;
    }}
'''
    
    # Add wall patches (bodies)
    for patch in wall_patches:
        u_content += f'''    {patch}
    {{
        type            noSlip;
    }}
'''
    
    u_content += '''}

// ************************************************************************* //
'''
    
    # Write p file with uniform initial condition  
    # Note: front/back are symmetry type in blockMeshDict (for 2D simulations)
    p_content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
    inlet
    {{
        type            zeroGradient;
    }}
    outlet
    {{
        type            fixedValue;
        value           uniform 0;
    }}
    top
    {{
        type            slip;
    }}
    bottom
    {{
        type            slip;
    }}
    front
    {{
        type            symmetry;
    }}
    back
    {{
        type            symmetry;
    }}
'''
    
    # Add wall patches (bodies)
    for patch in wall_patches:
        p_content += f'''    {patch}
    {{
        type            zeroGradient;
    }}
'''
    
    p_content += '''}

// ************************************************************************* //
'''
    
    # Write files
    with open(case_path / "0" / "U", 'w') as f:
        f.write(u_content)
    
    with open(case_path / "0" / "p", 'w') as f:
        f.write(p_content)


def run_meshing_workflow(case_path: Path, verbose: bool = True) -> bool:
    """
    Run the complete meshing workflow: blockMesh -> surfaceFeatureExtract -> snappyHexMesh
    Then reset initial conditions for the new mesh.
    """
    print("  [1/3] Running blockMesh...")
    if not run_openfoam_command(["blockMesh"], case_path, verbose=False):
        print("    ✗ blockMesh failed")
        return False
    print("    ✓ blockMesh complete")
    
    print("  [2/3] Running surfaceFeatureExtract...")
    if not run_openfoam_command(["surfaceFeatureExtract"], case_path, verbose=False):
        print("    ✗ surfaceFeatureExtract failed")
        return False
    print("    ✓ surfaceFeatureExtract complete")
    
    print("  [3/3] Running snappyHexMesh...")
    if not run_openfoam_command(["snappyHexMesh", "-overwrite"], case_path, verbose=False):
        print("    ✗ snappyHexMesh failed")
        return False
    print("    ✓ snappyHexMesh complete")
    
    # Reset initial conditions to uniform values for the new mesh
    print("  [4/4] Resetting initial conditions...")
    try:
        reset_initial_conditions(case_path, v_inf=10.0)
        print("    ✓ Initial conditions reset")
    except Exception as e:
        print(f"    ✗ Failed to reset initial conditions: {e}")
        return False
    
    return True


def run_potential_foam(case_path: Path, verbose: bool = True) -> bool:
    """Run potentialFoam solver with live output."""
    print("  Running potentialFoam...")
    
    # potentialFoam needs writeCellCentres for our comparison
    if not run_openfoam_command(["potentialFoam", "-writep"], case_path, verbose):
        print("    ✗ potentialFoam failed")
        return False
    print("    ✓ potentialFoam complete")
    
    # Write cell centres for comparison
    print("  Running postProcess -func writeCellCentres...")
    if not run_openfoam_command(["postProcess", "-func", "writeCellCentres"], case_path, verbose=False):
        print("    ✗ writeCellCentres failed")
        return False
    print("    ✓ Cell centres written")
    
    return True


def get_cell_count(case_path: Path) -> int:
    """Get the number of cells in the mesh using checkMesh."""
    result = subprocess.run(
        ["checkMesh", "-case", str(case_path)],
        capture_output=True,
        text=True
    )
    
    # Parse output for cell count
    for line in result.stdout.split('\n'):
        if 'cells:' in line:
            match = re.search(r'cells:\s*(\d+)', line)
            if match:
                return int(match.group(1))
    
    return 0


# =============================================================================
# Result Extraction
# =============================================================================

def extract_velocity_field(case_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract cell centres and velocity field from OpenFOAM case using foamlib.
    
    Returns:
        (cell_centres, velocity): Both as numpy arrays
    """
    if FOAMLIB_AVAILABLE:
        try:
            case = FoamCase(case_path)
            
            # Get latest time directory (for potentialFoam, results are in "0")
            latest_time = case[-1]
            
            # Read cell centres (written by postProcess -func writeCellCentres)
            C = np.array(latest_time["C"].internal_field)
            
            # Read velocity field
            U = np.array(latest_time["U"].internal_field)
            
            return C, U
        except Exception as e:
            print(f"    foamlib extraction failed: {e}")
            print(f"    Falling back to OpenFOAMRunner...")
    
    # Fallback: use our existing runner
    from validation import OpenFOAMRunner
    runner = OpenFOAMRunner(case_path, verbose=False)
    C = runner.get_cell_centres()
    U = runner.get_velocity_field()
    return C, U
    U = runner.get_velocity_field()
    return C, U


def create_comparison_grid(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create structured comparison grid from config."""
    grid_cfg = config['comparison_grid']
    
    x = np.linspace(grid_cfg['x_range'][0], grid_cfg['x_range'][1], grid_cfg['nx'])
    y = np.linspace(grid_cfg['y_range'][0], grid_cfg['y_range'][1], grid_cfg['ny'])
    XX, YY = np.meshgrid(x, y)
    points = np.column_stack([XX.ravel(), YY.ravel()])
    
    return XX, YY, points


def interpolate_to_grid(
    cell_centres: np.ndarray,
    velocity: np.ndarray,
    grid_points: np.ndarray,
    z_mid: float = 0.05
) -> np.ndarray:
    """
    Interpolate velocity field to comparison grid.
    
    Args:
        cell_centres: (N, 3) cell centre coordinates
        velocity: (N, 3) velocity vectors
        grid_points: (M, 2) comparison grid points
        z_mid: Z-coordinate of midplane
    
    Returns:
        (M, 2) interpolated velocity at grid points
    """
    # Filter to midplane
    mask = np.abs(cell_centres[:, 2] - z_mid) < 0.02
    C_2d = cell_centres[mask, :2]
    U_2d = velocity[mask, :2]
    
    # Interpolate
    Vx = griddata(C_2d, U_2d[:, 0], grid_points, method='linear')
    Vy = griddata(C_2d, U_2d[:, 1], grid_points, method='linear')
    
    # Handle NaN
    valid = ~(np.isnan(Vx) | np.isnan(Vy))
    
    result = np.column_stack([Vx, Vy])
    result[~valid] = 0  # Set invalid points to zero
    
    return result, valid


# =============================================================================
# Convergence Analysis
# =============================================================================

def calculate_gci(values: list[float], r: float = 1.5, p: float = None) -> dict:
    """
    Calculate Grid Convergence Index using Richardson extrapolation.
    
    Args:
        values: List of values from coarse to fine mesh (at least 3)
        r: Refinement ratio
        p: Order of convergence (calculated if None)
    
    Returns:
        dict with GCI metrics
    """
    if len(values) < 3:
        return {"error": "Need at least 3 refinement levels", "gci": None}
    
    # Get finest three values
    f3, f2, f1 = values[-3:]  # coarse, medium, fine
    
    # Calculate order of convergence if not provided
    if p is None:
        try:
            epsilon_32 = f3 - f2
            epsilon_21 = f2 - f1
            if abs(epsilon_21) < 1e-10:
                p = 2.0  # Assume second order if no change
            else:
                p = np.log(abs(epsilon_32 / epsilon_21)) / np.log(r)
                p = max(0.5, min(p, 4.0))  # Clamp to reasonable range
        except:
            p = 2.0
    
    # Richardson extrapolated value
    try:
        f_exact = f1 + (f1 - f2) / (r**p - 1)
    except:
        f_exact = f1
    
    # Grid Convergence Index
    Fs = 1.25  # Safety factor
    e21 = abs((f2 - f1) / f1) if abs(f1) > 1e-10 else 0
    gci_fine = Fs * e21 / (r**p - 1) if r**p > 1 else e21
    
    return {
        "order_of_convergence": p,
        "extrapolated_value": f_exact,
        "gci_fine": gci_fine,
        "relative_error_21": e21,
        "values": values
    }


# =============================================================================
# Main Convergence Study
# =============================================================================

def run_convergence_study(
    base_case_path: Path,
    output_dir: Path,
    config: dict,
    show_plots: bool = False
) -> dict:
    """
    Run the complete mesh convergence study.
    
    Args:
        base_case_path: Path to base OpenFOAM case
        output_dir: Directory to store results
        config: Configuration dictionary
        show_plots: Whether to display plots
    
    Returns:
        dict with convergence results
    """
    print("=" * 70)
    print("OpenFOAM Mesh Convergence Study")
    print("=" * 70)
    
    refinement_levels = config['refinement_levels']
    verbose = config['output'].get('verbose', True)
    delete_intermediate = config['output'].get('delete_intermediate_cases', True)
    
    # Create comparison grid
    XX, YY, grid_points = create_comparison_grid(config)
    print(f"\nComparison grid: {config['comparison_grid']['nx']}×{config['comparison_grid']['ny']} points")
    
    # Results storage
    results = {
        'levels': [],
        'cell_counts': [],
        'velocity_fields': [],
        'rms_velocity': [],
        'max_velocity': [],
        'mean_velocity': [],
        'rms_change': [],  # Change from previous level
        'config': config
    }
    
    prev_V_grid = None
    case_paths = []
    
    # Run each refinement level
    for i, level in enumerate(refinement_levels):
        print(f"\n{'='*70}")
        print(f"Level {i+1}/{len(refinement_levels)}: {level['name']}")
        print(f"  blockMesh: {level['blockMesh_cells']}")
        print(f"  snappy surface: {level['snappy_surface_level']}, feature: {level['snappy_feature_level']}")
        print("=" * 70)
        
        # Create case directory
        case_name = f"level_{i+1}_{level['name']}"
        case_path = output_dir / "cases" / case_name
        case_paths.append(case_path)
        
        # Copy base case
        if case_path.exists():
            shutil.rmtree(case_path)
        shutil.copytree(base_case_path, case_path)
        
        # Modify mesh parameters
        print("\nModifying mesh parameters...")
        modify_blockmesh_dict(case_path, level['blockMesh_cells'])
        modify_snappy_dict(case_path, level['snappy_surface_level'], level['snappy_feature_level'])
        
        # Run meshing
        print("\nRunning meshing workflow...")
        t0 = time.time()
        if not run_meshing_workflow(case_path, verbose):
            print(f"  ✗ Meshing failed for level {level['name']}")
            continue
        mesh_time = time.time() - t0
        
        # Get cell count
        cell_count = get_cell_count(case_path)
        print(f"\n  Mesh cells: {cell_count:,}")
        print(f"  Meshing time: {mesh_time:.1f}s")
        
        # Run solver
        print("\nRunning solver...")
        t0 = time.time()
        if not run_potential_foam(case_path, verbose):
            print(f"  ✗ Solver failed for level {level['name']}")
            continue
        solve_time = time.time() - t0
        print(f"  Solve time: {solve_time:.1f}s")
        
        # Extract results
        print("\nExtracting velocity field...")
        try:
            C, U = extract_velocity_field(case_path)
            V_grid, valid = interpolate_to_grid(C, U, grid_points)
            V_mag = np.sqrt((V_grid**2).sum(axis=1))
            
            # Store results
            results['levels'].append(level['name'])
            results['cell_counts'].append(cell_count)
            results['velocity_fields'].append(V_grid)
            results['rms_velocity'].append(np.sqrt(np.mean(V_mag[valid]**2)))
            results['max_velocity'].append(V_mag[valid].max())
            results['mean_velocity'].append(V_mag[valid].mean())
            
            # Calculate change from previous level
            if prev_V_grid is not None:
                diff = V_grid - prev_V_grid
                rms_change = np.sqrt(np.mean(diff[valid]**2))
                results['rms_change'].append(rms_change)
                print(f"  RMS change from previous: {rms_change:.4f} m/s")
            else:
                results['rms_change'].append(None)
            
            prev_V_grid = V_grid.copy()
            
            print(f"  |V| range: [{V_mag[valid].min():.2f}, {V_mag[valid].max():.2f}] m/s")
            print(f"  |V| mean: {V_mag[valid].mean():.2f} m/s")
            
        except Exception as e:
            print(f"  ✗ Error extracting results: {e}")
            continue
    
    # Calculate GCI
    print("\n" + "=" * 70)
    print("Convergence Analysis")
    print("=" * 70)
    
    if len(results['rms_velocity']) >= 3:
        gci_results = calculate_gci(
            results['mean_velocity'],
            r=config['convergence']['refinement_ratio']
        )
        results['gci'] = gci_results
        
        print(f"\nRichardson Extrapolation Results:")
        print(f"  Order of convergence: {gci_results['order_of_convergence']:.2f}")
        print(f"  Extrapolated value: {gci_results['extrapolated_value']:.4f} m/s")
        print(f"  GCI (fine mesh): {gci_results['gci_fine']*100:.2f}%")
        
        if gci_results['gci_fine'] < config['convergence']['gci_threshold']:
            print(f"  ✓ Converged! (GCI < {config['convergence']['gci_threshold']*100:.0f}%)")
        else:
            print(f"  ✗ Not converged (GCI > {config['convergence']['gci_threshold']*100:.0f}%)")
    else:
        results['gci'] = None
        print("  Need at least 3 successful levels for GCI calculation")
    
    # Save final case
    if config['output'].get('save_final_case', True) and case_paths:
        final_case_path = base_case_path.parent / "final_case"
        if final_case_path.exists():
            shutil.rmtree(final_case_path)
        shutil.copytree(case_paths[-1], final_case_path)
        print(f"\n✓ Final case saved to: {final_case_path}")
        results['final_case_path'] = str(final_case_path)
    
    # Clean up intermediate cases
    if delete_intermediate and len(case_paths) > 1:
        print("\nCleaning up intermediate cases...")
        for case_path in case_paths[:-1]:  # Keep the last one
            if case_path.exists():
                shutil.rmtree(case_path)
                print(f"  Deleted: {case_path.name}")
    
    # Plot results
    plot_convergence_results(results, output_dir, show=show_plots)
    
    # Save results
    save_results(results, output_dir)
    
    return results


def plot_convergence_results(results: dict, output_dir: Path, show: bool = False) -> None:
    """Plot mesh convergence results."""
    if len(results['cell_counts']) < 2:
        print("Not enough data points for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    cells = np.array(results['cell_counts'])
    
    # 1. Mean velocity vs cell count
    ax = axes[0, 0]
    ax.semilogx(cells, results['mean_velocity'], 'o-', linewidth=2, markersize=8)
    if results.get('gci') and results['gci'].get('extrapolated_value'):
        ax.axhline(results['gci']['extrapolated_value'], color='r', linestyle='--', 
                   label=f"Extrapolated: {results['gci']['extrapolated_value']:.3f}")
        ax.legend()
    ax.set_xlabel('Number of Cells')
    ax.set_ylabel('Mean Velocity Magnitude (m/s)')
    ax.set_title('Mean Velocity Convergence')
    ax.grid(True, alpha=0.3)
    
    # 2. RMS change vs cell count
    ax = axes[0, 1]
    rms_changes = [r for r in results['rms_change'] if r is not None]
    if rms_changes:
        ax.loglog(cells[1:], rms_changes, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Cells')
        ax.set_ylabel('RMS Change from Previous Level (m/s)')
        ax.set_title('Solution Change Between Levels')
        ax.grid(True, alpha=0.3, which='both')
    
    # 3. Max velocity vs cell count
    ax = axes[1, 0]
    ax.semilogx(cells, results['max_velocity'], 's-', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('Number of Cells')
    ax.set_ylabel('Max Velocity Magnitude (m/s)')
    ax.set_title('Maximum Velocity Convergence')
    ax.grid(True, alpha=0.3)
    
    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    table_data = []
    for i, level in enumerate(results['levels']):
        row = [
            level,
            f"{results['cell_counts'][i]:,}",
            f"{results['mean_velocity'][i]:.3f}",
            f"{results['rms_change'][i]:.4f}" if results['rms_change'][i] else "-"
        ]
        table_data.append(row)
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Level', 'Cells', 'Mean |V| (m/s)', 'RMS Change'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Add GCI info
    if results.get('gci') and results['gci'].get('gci_fine'):
        gci = results['gci']
        info_text = (
            f"GCI Analysis:\n"
            f"Order of convergence: {gci['order_of_convergence']:.2f}\n"
            f"Extrapolated value: {gci['extrapolated_value']:.4f} m/s\n"
            f"GCI (fine): {gci['gci_fine']*100:.2f}%"
        )
        ax.text(0.5, 0.15, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('OpenFOAM Mesh Convergence Study', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    plot_path = output_dir / "mesh_convergence.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {plot_path}")
    
    if show:
        plt.show()
    plt.close()


def save_results(results: dict, output_dir: Path) -> None:
    """Save convergence results to files."""
    import csv
    
    # Save CSV
    csv_path = output_dir / "convergence_data.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['level', 'cells', 'mean_velocity', 'max_velocity', 'rms_change'])
        for i in range(len(results['levels'])):
            writer.writerow([
                results['levels'][i],
                results['cell_counts'][i],
                results['mean_velocity'][i],
                results['max_velocity'][i],
                results['rms_change'][i] if results['rms_change'][i] else ''
            ])
    print(f"✓ CSV saved: {csv_path}")
    
    # Save summary YAML
    summary = {
        'levels': results['levels'],
        'cell_counts': results['cell_counts'],
        'mean_velocity': [float(v) for v in results['mean_velocity']],
        'max_velocity': [float(v) for v in results['max_velocity']],
        'rms_change': [float(v) if v else None for v in results['rms_change']],
        'gci': results.get('gci'),
        'final_case_path': results.get('final_case_path')
    }
    
    yaml_path = output_dir / "convergence_summary.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    print(f"✓ Summary saved: {yaml_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OpenFOAM mesh convergence study for validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python openfoam_convergence.py cases/two_rounded_rects
    python openfoam_convergence.py cases/two_rounded_rects --show
    python openfoam_convergence.py cases/two_rounded_rects --create-config-only
        """
    )
    parser.add_argument("case_path", help="Path to the panel method case (e.g., cases/two_rounded_rects)")
    parser.add_argument("--show", action="store_true", help="Display plots")
    parser.add_argument("--create-config-only", action="store_true", 
                        help="Only create default config file, don't run study")
    args = parser.parse_args()
    
    # Resolve paths
    case_path = Path(args.case_path)
    case_name = case_path.name
    
    # Directory structure
    validation_dir = Path("validation_results") / case_name
    base_openfoam_case = validation_dir / "openfoam"
    convergence_dir = validation_dir / "openfoam_convergence"
    config_path = convergence_dir / "config.yaml"
    
    # Check base case exists
    if not base_openfoam_case.exists():
        print(f"ERROR: OpenFOAM case not found at: {base_openfoam_case}")
        print("Run validation first to create the OpenFOAM case:")
        print(f"  python validation/scripts/run_validation.py --run-openfoam {case_path}")
        sys.exit(1)
    
    # Create output directory
    convergence_dir.mkdir(parents=True, exist_ok=True)
    
    # Create or load config
    if not config_path.exists():
        create_default_config(config_path)
        print(f"\nEdit the config file as needed, then re-run:")
        print(f"  python {sys.argv[0]} {args.case_path}")
        
        if args.create_config_only:
            sys.exit(0)
    
    if args.create_config_only:
        print(f"Config file already exists: {config_path}")
        sys.exit(0)
    
    # Load config
    config = load_config(config_path)
    print(f"\nLoaded config from: {config_path}")
    print(f"Refinement levels: {len(config['refinement_levels'])}")
    
    # Run study
    results = run_convergence_study(
        base_openfoam_case,
        convergence_dir,
        config,
        show_plots=args.show
    )
    
    print("\n" + "=" * 70)
    print("✓ Mesh Convergence Study Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {convergence_dir}")
    if results.get('final_case_path'):
        print(f"Final case at: {results['final_case_path']}")
    print("\nNext steps:")
    print(f"  1. Review results in {convergence_dir}/mesh_convergence.png")
    print(f"  2. If not converged, add finer levels to {config_path}")
    print(f"  3. Run panel method comparison against final_case")


if __name__ == "__main__":
    main()
