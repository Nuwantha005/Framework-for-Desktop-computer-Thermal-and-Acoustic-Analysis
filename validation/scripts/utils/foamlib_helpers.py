"""
Foamlib utilities for modifying OpenFOAM case files.

Uses foamlib's FoamCase API for structured editing of OpenFOAM dictionaries.
Reference: notes/AI/lmArena/ChatGPT/foamLib_Useage.md

For running OpenFOAM workflows, this module delegates to OpenFOAMRunner
which provides additional features like parallel snappyHexMesh execution.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from foamlib import FoamCase

# Import runner for workflow execution
from validation.adapters.openfoam import OpenFOAMRunner


def set_blockmesh_cells(case_dir: Path, cells: Tuple[int, int, int]) -> None:
    """
    Modify blockMeshDict to set cell counts using foamlib FoamCase.
    
    Args:
        case_dir: OpenFOAM case directory
        cells: (nx, ny, nz) cell counts
    """
    nx, ny, nz = cells
    case = FoamCase(case_dir)
    
    with case.block_mesh_dict as f:
        # blocks is a token list: ["hex", [vertices], [nx,ny,nz], "simpleGrading", [1,1,1]]
        blocks = list(f["blocks"])
        blocks[2] = [nx, ny, nz]  # cell counts at position 2
        f["blocks"] = blocks


def set_snappy_levels(
    case_dir: Path,
    surface_level: int,
    feature_level: int
) -> None:
    """
    Modify snappyHexMeshDict to set refinement levels for all components.
    
    Args:
        case_dir: OpenFOAM case directory
        surface_level: Surface refinement level
        feature_level: Feature edge refinement level
    """
    case = FoamCase(case_dir)
    snappy = case.file("system/snappyHexMeshDict")
    
    with snappy as f:
        # Update all feature entries (list of dicts)
        features = list(f["castellatedMeshControls", "features"])
        for entry in features:
            entry["level"] = feature_level
        f["castellatedMeshControls", "features"] = features
        
        # Update all refinementSurfaces entries (dict keyed by component name)
        ref = dict(f["castellatedMeshControls", "refinementSurfaces"])
        for name, sub in ref.items():
            sub["level"] = [surface_level, surface_level]
        f["castellatedMeshControls", "refinementSurfaces"] = ref


def set_snappy_levels_per_component(
    case_dir: Path,
    component_levels: Dict[str, Dict[str, int]]
) -> None:
    """
    Modify snappyHexMeshDict to set per-component refinement levels.
    
    Args:
        case_dir: OpenFOAM case directory
        component_levels: Dict[component_name] = {"surface_level": N, "feature_level": M}
    """
    case = FoamCase(case_dir)
    snappy = case.file("system/snappyHexMeshDict")
    
    with snappy as f:
        # Update features - match by component name in filename
        features = list(f["castellatedMeshControls", "features"])
        for entry in features:
            # Feature file is like "ComponentName.eMesh"
            filename = entry.get("file", "")
            comp_name = filename.replace(".eMesh", "")
            if comp_name in component_levels:
                entry["level"] = component_levels[comp_name].get("feature_level", 2)
        f["castellatedMeshControls", "features"] = features
        
        # Update refinementSurfaces
        ref = dict(f["castellatedMeshControls", "refinementSurfaces"])
        for name, sub in ref.items():
            if name in component_levels:
                level = component_levels[name].get("surface_level", 2)
                sub["level"] = [level, level]
        f["castellatedMeshControls", "refinementSurfaces"] = ref


def set_blockmesh_domain(
    case_dir: Path,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    cells: Tuple[int, int, int],
) -> None:
    """
    Set full blockMeshDict domain: vertices, blocks, and boundary patches.
    
    Args:
        case_dir: OpenFOAM case directory
        x_range: (x0, x1) domain bounds
        y_range: (y0, y1) domain bounds
        z_range: (z0, z1) domain bounds
        cells: (nx, ny, nz) cell counts
    """
    x0, x1 = x_range
    y0, y1 = y_range
    z0, z1 = z_range
    nx, ny, nz = cells
    
    case = FoamCase(case_dir)
    
    with case.block_mesh_dict as f:
        f["scale"] = 1
        
        # Vertices: 8 corners of the box
        f["vertices"] = [
            [x0, y0, z0],  # 0
            [x1, y0, z0],  # 1
            [x1, y1, z0],  # 2
            [x0, y1, z0],  # 3
            [x0, y0, z1],  # 4
            [x1, y0, z1],  # 5
            [x1, y1, z1],  # 6
            [x0, y1, z1],  # 7
        ]
        
        # Blocks: single hex block
        f["blocks"] = [
            "hex",
            [0, 1, 2, 3, 4, 5, 6, 7],
            [nx, ny, nz],
            "simpleGrading",
            [1, 1, 1],
        ]
        
        f["edges"] = []
        
        # Boundary patches for 2D case (empty front/back)
        f["boundary"] = [
            ("inlet",  {"type": "patch", "faces": [[0, 4, 7, 3]]}),
            ("outlet", {"type": "patch", "faces": [[1, 2, 6, 5]]}),
            ("top",    {"type": "patch", "faces": [[3, 7, 6, 2]]}),
            ("bottom", {"type": "patch", "faces": [[0, 1, 5, 4]]}),
            ("front",  {"type": "empty", "faces": [[0, 3, 2, 1]]}),
            ("back",   {"type": "empty", "faces": [[4, 5, 6, 7]]}),
        ]
        
        f["mergePatchPairs"] = []


def run_openfoam_workflow(
    case_dir: Path,
    verbose: bool = True,
    parallel_snappy: bool = False,
    n_procs: int = 4,
    solver: str = "potentialFoam"
) -> bool:
    """
    Run OpenFOAM meshing and solving workflow using OpenFOAMRunner.
    
    Steps:
        1. blockMesh
        2. surfaceFeatureExtract
        3. snappyHexMesh -overwrite (optionally in parallel)
        4. extrudeMesh
        5. potentialFoam -writep (or specified solver)
        6. writeCellCentres (for comparison)
    
    Args:
        case_dir: OpenFOAM case directory
        verbose: Print progress
        parallel_snappy: If True, run snappyHexMesh in parallel using MPI
            (useful for finer meshes in convergence studies)
        n_procs: Number of MPI processes for parallel snappy (default: 4)
        solver: Solver to run (default: potentialFoam)
    
    Returns:
        True if successful
    """
    try:
        runner = OpenFOAMRunner(case_dir, verbose=verbose)
        success = runner.run_all(
            solver=solver,
            use_snappy=True,
            parallel_snappy=parallel_snappy,
            n_procs=n_procs,
            use_extrude=True
        )
        return success
        
    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return False


def get_latest_time_dir(case_dir: Path) -> Optional[Path]:
    """Get the latest time directory in the case."""
    case = FoamCase(case_dir)
    times = list(case)
    if times:
        return case.path / str(times[-1].time)
    return None


def run_write_cell_centres(case_dir: Path) -> bool:
    """Run writeCellCentres utility."""
    case = FoamCase(case_dir)
    try:
        case.run("writeCellCentres")
        return True
    except Exception as e:
        print(f"  Error running writeCellCentres: {e}")
        return False


def run_post_process(case_dir: Path) -> bool:
    """Run postProcess utility."""
    case = FoamCase(case_dir)
    try:
        case.run(["postProcess", "-fields", "(U p)"])
        return True
    except Exception as e:
        print(f"  Error running postProcess: {e}")
        return False
