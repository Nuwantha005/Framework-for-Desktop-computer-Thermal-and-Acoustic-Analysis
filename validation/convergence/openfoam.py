"""
OpenFOAM mesh convergence study utilities.

Provides functions for running OpenFOAM at multiple mesh resolutions,
extracting fields, and computing convergence metrics.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import shutil
import yaml
import time
import numpy as np
from numpy.typing import NDArray

from validation.adapters.openfoam.runner import OpenFOAMRunner
from validation.adapters.openfoam.case_generator import OpenFOAMCaseGenerator, MeshSettings


@dataclass
class OpenFOAMMeshConfig:
    """Configuration for a single mesh level in convergence study."""
    name: str
    background_cells_per_unit: float
    refinement_level: int
    z_thickness: float = 0.1
    
    def __str__(self) -> str:
        return f"{self.name}: bg={self.background_cells_per_unit:.1f}, ref={self.refinement_level}"


@dataclass
class OpenFOAMConvergenceResult:
    """Result from a single OpenFOAM mesh level."""
    config: OpenFOAMMeshConfig
    case_dir: Path
    num_cells: int
    solve_time: float
    mesh_time: float
    success: bool
    error_msg: Optional[str] = None
    
    # Field data
    cell_centres: Optional[NDArray] = None  # (N, 3)
    velocity_field: Optional[NDArray] = None  # (N, 3)
    pressure_field: Optional[NDArray] = None  # (N,)
    
    @property
    def total_time(self) -> float:
        """Total wall-clock time (mesh + solve)."""
        return self.mesh_time + self.solve_time


def create_mesh_configs(
    base_density: float = 5.0,
    num_levels: int = 4,
    refinement_ratio: float = 1.5
) -> List[OpenFOAMMeshConfig]:
    """
    Create a series of mesh configurations with increasing density.
    
    Args:
        base_density: Coarsest mesh density (cells per unit length)
        num_levels: Number of mesh levels
        refinement_ratio: Ratio between successive meshes
    
    Returns:
        List of mesh configurations from coarse to fine
    
    Examples:
        >>> configs = create_mesh_configs(base_density=5.0, num_levels=3)
        >>> for cfg in configs:
        ...     print(cfg)
        coarse: bg=5.0, ref=1
        medium: bg=7.5, ref=2
        fine: bg=11.2, ref=2
    """
    configs = []
    level_names = ["coarse", "medium", "fine", "finest", "ultra-fine"]
    
    for i in range(num_levels):
        density = base_density * (refinement_ratio ** i)
        # Increase refinement level for finer meshes (stay at 2 or 3)
        ref_level = min(2 + (i // 2), 3)
        
        name = level_names[i] if i < len(level_names) else f"level_{i}"
        
        configs.append(OpenFOAMMeshConfig(
            name=name,
            background_cells_per_unit=density,
            refinement_level=ref_level
        ))
    
    return configs


def run_openfoam_convergence(
    case,
    output_dir: Path,
    mesh_configs: List[OpenFOAMMeshConfig],
    solver: str = "potentialFoam",
    use_snappy: bool = True,
    domain_padding: float = 2.0,
    clean_on_success: bool = True,
    verbose: bool = True
) -> List[OpenFOAMConvergenceResult]:
    """
    Run OpenFOAM mesh convergence study.
    
    Generates and runs OpenFOAM cases for each mesh configuration,
    extracts fields, and returns results for analysis.
    
    Args:
        case: Panel method case (from CaseLoader)
        output_dir: Directory for convergence study results
        mesh_configs: List of mesh configurations to test
        solver: OpenFOAM solver to use
        use_snappy: Whether to use snappyHexMesh
        domain_padding: Padding around geometry (in characteristic lengths)
        clean_on_success: Clean intermediate mesh files to save space
        verbose: Print progress messages
    
    Returns:
        List of OpenFOAMConvergenceResult for each mesh level
    
    Examples:
        >>> from core.io import CaseLoader
        >>> case = CaseLoader.load_case("cases/single_square")
        >>> configs = create_mesh_configs(5.0, 3)
        >>> results = run_openfoam_convergence(
        ...     case, Path("validation_results/conv_study"), configs
        ... )
        >>> for res in results:
        ...     print(f"{res.config.name}: {res.num_cells} cells, {res.solve_time:.1f}s")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_file = output_dir / "convergence_config.yaml"
    config_data = {
        'case_name': case.name,
        'solver': solver,
        'use_snappy': use_snappy,
        'domain_padding': domain_padding,
        'mesh_levels': [
            {
                'name': cfg.name,
                'background_cells_per_unit': cfg.background_cells_per_unit,
                'refinement_level': cfg.refinement_level,
                'z_thickness': cfg.z_thickness
            }
            for cfg in mesh_configs
        ]
    }
    
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)
    
    if verbose:
        print("=" * 70)
        print("OpenFOAM Mesh Convergence Study")
        print("=" * 70)
        print(f"Case: {case.name}")
        print(f"Mesh levels: {len(mesh_configs)}")
        print(f"Output: {output_dir}")
        print()
    
    results = []
    
    for i, mesh_cfg in enumerate(mesh_configs, 1):
        if verbose:
            print(f"\n--- Mesh Level {i}/{len(mesh_configs)}: {mesh_cfg.name} ---")
            print(f"    {mesh_cfg}")
        
        case_dir = output_dir / f"case_{mesh_cfg.name}"
        
        try:
            # Generate case
            t0 = time.time()
            
            mesh_settings = MeshSettings(
                background_cells_per_unit=mesh_cfg.background_cells_per_unit,
                refinement_level=mesh_cfg.refinement_level,
                z_thickness=mesh_cfg.z_thickness
            )
            
            generator = OpenFOAMCaseGenerator(
                case=case,
                output_dir=case_dir,
                solver_type=solver,
                mesh_settings=mesh_settings,
                domain_padding=domain_padding
            )
            
            of_case_dir = generator.generate()
            
            # Run OpenFOAM
            runner = OpenFOAMRunner(of_case_dir, verbose=False)
            
            mesh_start = time.time()
            success = runner.run_all(solver=solver, use_snappy=use_snappy)
            solve_end = time.time()
            
            mesh_time = mesh_start - t0
            solve_time = solve_end - mesh_start
            
            if not success:
                results.append(OpenFOAMConvergenceResult(
                    config=mesh_cfg,
                    case_dir=of_case_dir,
                    num_cells=0,
                    solve_time=solve_time,
                    mesh_time=mesh_time,
                    success=False,
                    error_msg="OpenFOAM run failed (see logs)"
                ))
                if verbose:
                    print(f"    ✗ FAILED")
                continue
            
            # Extract fields
            C = runner.get_cell_centres()
            U = runner.get_velocity_field()
            p = runner.get_pressure_field()
            
            num_cells = len(C)
            
            if verbose:
                print(f"    ✓ Success: {num_cells} cells")
                print(f"    Mesh time: {mesh_time:.1f}s, Solve time: {solve_time:.1f}s")
            
            # Store result
            results.append(OpenFOAMConvergenceResult(
                config=mesh_cfg,
                case_dir=of_case_dir,
                num_cells=num_cells,
                solve_time=solve_time,
                mesh_time=mesh_time,
                success=True,
                cell_centres=C,
                velocity_field=U,
                pressure_field=p
            ))
            
            # Clean up to save space
            if clean_on_success:
                _clean_case(of_case_dir, keep_results=True)
        
        except Exception as e:
            if verbose:
                print(f"    ✗ ERROR: {e}")
            
            results.append(OpenFOAMConvergenceResult(
                config=mesh_cfg,
                case_dir=case_dir,
                num_cells=0,
                solve_time=0.0,
                mesh_time=0.0,
                success=False,
                error_msg=str(e)
            ))
    
    if verbose:
        print("\n" + "=" * 70)
        print("Convergence study complete!")
        print(f"Successful runs: {sum(r.success for r in results)}/{len(results)}")
        print("=" * 70)
    
    return results


def extract_openfoam_fields(
    case_dir: Path,
    time_idx: int = -1
) -> Dict[str, NDArray]:
    """
    Extract OpenFOAM fields from a case directory.
    
    Args:
        case_dir: Path to OpenFOAM case
        time_idx: Time index (-1 for latest)
    
    Returns:
        Dictionary with 'C', 'U', 'p' arrays
    
    Examples:
        >>> fields = extract_openfoam_fields(Path("validation_results/case"))
        >>> print(f"Cells: {len(fields['C'])}")
        >>> print(f"Velocity range: {fields['U'][:, 0].min():.2f} to {fields['U'][:, 0].max():.2f}")
    """
    runner = OpenFOAMRunner(case_dir, verbose=False)
    
    C = runner.get_cell_centres(time_idx)
    U = runner.get_velocity_field(time_idx)
    p = runner.get_pressure_field(time_idx)
    
    return {
        'C': C,
        'U': U,
        'p': p
    }


def interpolate_openfoam_to_grid(
    cell_centres: NDArray,
    velocity_field: NDArray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    resolution: Tuple[int, int],
    z_plane: float = 0.05,
    z_tolerance: float = 0.02
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Interpolate OpenFOAM 3D results to a 2D structured grid.
    
    Args:
        cell_centres: (N, 3) array of cell centers
        velocity_field: (N, 3) array of velocities
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        resolution: (nx, ny)
        z_plane: Z coordinate for 2D slice
        z_tolerance: Tolerance for selecting cells near z_plane
    
    Returns:
        (XX, YY, Vx, Vy) structured grid arrays
    
    Examples:
        >>> C = np.random.rand(1000, 3) * 10
        >>> U = np.random.rand(1000, 3)
        >>> XX, YY, Vx, Vy = interpolate_openfoam_to_grid(
        ...     C, U, (-5, 5), (-5, 5), (50, 50)
        ... )
    """
    from scipy.interpolate import griddata
    
    # Filter to z-plane
    mask = np.abs(cell_centres[:, 2] - z_plane) < z_tolerance
    
    if mask.sum() < 10:
        raise ValueError(
            f"Only {mask.sum()} cells near z={z_plane}. "
            f"Check z_plane value or increase z_tolerance."
        )
    
    points_2d = cell_centres[mask, :2]
    U_2d = velocity_field[mask, :2]
    
    # Create structured grid
    nx, ny = resolution
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    XX, YY = np.meshgrid(x, y)
    
    # Interpolate
    Vx = griddata(points_2d, U_2d[:, 0], (XX, YY), method='linear')
    Vy = griddata(points_2d, U_2d[:, 1], (XX, YY), method='linear')
    
    return XX, YY, Vx, Vy


def _clean_case(case_dir: Path, keep_results: bool = True):
    """
    Clean OpenFOAM case to save disk space.
    
    Args:
        case_dir: OpenFOAM case directory
        keep_results: If True, keep latest time directory
    """
    # Remove processor directories (parallel runs)
    for proc_dir in case_dir.glob("processor*"):
        if proc_dir.is_dir():
            shutil.rmtree(proc_dir)
    
    # Remove intermediate time directories
    if keep_results:
        from foamlib import FoamCase
        foam_case = FoamCase(case_dir)
        times = list(foam_case)
        
        if len(times) > 1:
            # Keep only latest
            for time_dir in times[:-1]:
                time_path = case_dir / str(time_dir.time)
                if time_path.exists():
                    shutil.rmtree(time_path)
    
    # Remove polyMesh backup directories
    poly_mesh = case_dir / "constant" / "polyMesh"
    if poly_mesh.exists():
        for backup in poly_mesh.glob("*_*"):
            if backup.is_dir():
                shutil.rmtree(backup)


def compare_openfoam_meshes(
    results: List[OpenFOAMConvergenceResult],
    quantity: str = "velocity_magnitude"
) -> Dict[str, NDArray]:
    """
    Compare a quantity across different OpenFOAM mesh levels.
    
    Args:
        results: List of OpenFOAM convergence results
        quantity: Quantity to compare ("velocity_magnitude", "pressure")
    
    Returns:
        Dictionary with mesh names as keys and mean values
    
    Examples:
        >>> results = [...]  # from run_openfoam_convergence
        >>> comparison = compare_openfoam_meshes(results, "velocity_magnitude")
        >>> for name, value in comparison.items():
        ...     print(f"{name}: {value:.4f} m/s")
    """
    comparison = {}
    
    for result in results:
        if not result.success:
            continue
        
        if quantity == "velocity_magnitude":
            U = result.velocity_field
            value = np.mean(np.sqrt((U**2).sum(axis=1)))
        elif quantity == "pressure":
            value = np.mean(result.pressure_field)
        else:
            raise ValueError(f"Unknown quantity: {quantity}")
        
        comparison[result.config.name] = value
    
    return comparison
