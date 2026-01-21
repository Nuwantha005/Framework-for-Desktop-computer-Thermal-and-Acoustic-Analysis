"""
OpenFOAM runner using foamlib.

Wraps foamlib.FoamCase for executing OpenFOAM commands.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import subprocess
import time

from foamlib import FoamCase, FoamFieldFile


@dataclass
class RunResult:
    """Result of an OpenFOAM command execution."""
    command: str
    success: bool
    return_code: int
    stdout: str
    stderr: str
    runtime: float  # seconds
    log_file: Optional[Path] = None
    
    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"[{status}] {self.command} (rc={self.return_code}, {self.runtime:.1f}s)"


class OpenFOAMRunner:
    """
    Execute OpenFOAM commands for a case using foamlib.
    
    Handles:
    - Mesh generation (blockMesh, snappyHexMesh)
    - Solver execution (potentialFoam, simpleFoam, etc.)
    - Post-processing (sample, postProcess)
    - Result access via foamlib
    
    Usage:
        runner = OpenFOAMRunner("validation_results/my_case")
        
        # Run full workflow
        runner.run_all()
        
        # Or run steps individually
        runner.run_blockmesh()
        runner.run_snappy()
        runner.run_solver("potentialFoam")
        
        # Access results via foamlib
        case = runner.foam_case
        latest = case[-1]  # Latest time directory
        U = latest["U"].internal_field
        p = latest["p"].internal_field
    """
    
    def __init__(
        self,
        case_dir: Path | str,
        verbose: bool = True
    ):
        """
        Initialize runner.
        
        Args:
            case_dir: Path to OpenFOAM case directory
            verbose: If True, print progress messages
        """
        self.case_dir = Path(case_dir).resolve()
        self.verbose = verbose
        self.results: List[RunResult] = []
        
        if not self.case_dir.exists():
            raise FileNotFoundError(f"Case directory not found: {self.case_dir}")
        
        # Create foamlib FoamCase wrapper
        self._foam_case = FoamCase(self.case_dir)
        
        # Verify OpenFOAM is available
        self._check_openfoam()
    
    @property
    def foam_case(self) -> FoamCase:
        """Access the foamlib FoamCase object for result reading."""
        return self._foam_case
    
    def _check_openfoam(self):
        """Check if OpenFOAM commands are available."""
        try:
            result = subprocess.run(
                ["blockMesh", "-help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if self.verbose:
                print("✓ OpenFOAM found in PATH")
        except FileNotFoundError:
            raise RuntimeError(
                "OpenFOAM not found. Please source OpenFOAM environment:\n"
                "  source /opt/openfoam/etc/bashrc"
            )
    
    def _run_command(
        self,
        command: str | List[str],
        log_name: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> RunResult:
        """
        Run an OpenFOAM command.
        
        Args:
            command: Command to run (string or list)
            log_name: Name for log file (e.g., "blockMesh" -> log.blockMesh)
            timeout: Timeout in seconds
        
        Returns:
            RunResult with command outcome
        """
        if isinstance(command, str):
            cmd_str = command
        else:
            cmd_str = " ".join(command)
        
        if self.verbose:
            print(f"  Running: {cmd_str}...")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd_str,
                shell=True,
                cwd=self.case_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                executable="/bin/bash"
            )
            success = result.returncode == 0
            stdout = result.stdout
            stderr = result.stderr
            return_code = result.returncode
            
        except subprocess.TimeoutExpired as e:
            success = False
            stdout = e.stdout.decode() if e.stdout else ""
            stderr = f"Command timed out after {timeout}s"
            return_code = -1
            
        except Exception as e:
            success = False
            stdout = ""
            stderr = str(e)
            return_code = -1
        
        runtime = time.time() - start_time
        
        # Write log file
        log_file = None
        if log_name:
            log_file = self.case_dir / f"log.{log_name}"
            with open(log_file, 'w') as f:
                f.write(f"Command: {cmd_str}\n")
                f.write(f"Return code: {return_code}\n")
                f.write(f"Runtime: {runtime:.2f}s\n")
                f.write("\n=== STDOUT ===\n")
                f.write(stdout)
                f.write("\n=== STDERR ===\n")
                f.write(stderr)
        
        run_result = RunResult(
            command=cmd_str,
            success=success,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            runtime=runtime,
            log_file=log_file
        )
        
        self.results.append(run_result)
        
        if self.verbose:
            print(f"    {run_result}")
        
        return run_result
    
    def run_blockmesh(self, timeout: float = 300) -> RunResult:
        """Run blockMesh to create background mesh."""
        return self._run_command("blockMesh", "blockMesh", timeout)
    
    def run_surface_feature_extract(self, timeout: float = 300) -> RunResult:
        """Run surfaceFeatureExtract for snappyHexMesh."""
        return self._run_command("surfaceFeatureExtract", "surfaceFeatureExtract", timeout)
    
    def run_snappy(self, overwrite: bool = True, timeout: float = 1800) -> RunResult:
        """
        Run snappyHexMesh to refine mesh around bodies.
        
        Args:
            overwrite: If True, overwrite existing mesh
            timeout: Timeout in seconds (default 30 min)
        """
        cmd = "snappyHexMesh"
        if overwrite:
            cmd += " -overwrite"
        return self._run_command(cmd, "snappyHexMesh", timeout)
    
    def run_check_mesh(self, timeout: float = 300) -> RunResult:
        """Run checkMesh to verify mesh quality."""
        return self._run_command("checkMesh", "checkMesh", timeout)
    
    def run_solver(self, solver: str = "potentialFoam", timeout: float = 3600) -> RunResult:
        """
        Run OpenFOAM solver.
        
        Args:
            solver: Solver name (potentialFoam, simpleFoam, etc.)
            timeout: Timeout in seconds (default 1 hour)
        """
        return self._run_command(solver, solver, timeout)
    
    def run_write_cell_centres(self, timeout: float = 300) -> RunResult:
        """Run postProcess to write cell centre coordinates (needed for comparison)."""
        return self._run_command("postProcess -func writeCellCentres", "writeCellCentres", timeout)
    
    def run_sample(self, timeout: float = 300) -> RunResult:
        """Run postProcess to sample fields."""
        return self._run_command("postProcess -func sample", "sample", timeout)
    
    def run_all(
        self,
        solver: str = "potentialFoam",
        use_snappy: bool = True
    ) -> bool:
        """
        Run complete workflow: mesh → solve → sample.
        
        Args:
            solver: Solver to use
            use_snappy: If True, use snappyHexMesh for body refinement
        
        Returns:
            True if all steps succeeded
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Running OpenFOAM workflow in: {self.case_dir}")
            print(f"{'='*60}\n")
        
        # Meshing
        result = self.run_blockmesh()
        if not result.success:
            print(f"ERROR: blockMesh failed")
            return False
        
        if use_snappy:
            # Check if STL files exist
            stl_dir = self.case_dir / "constant" / "triSurface"
            stl_files = list(stl_dir.glob("*.stl")) if stl_dir.exists() else []
            
            if stl_files:
                result = self.run_surface_feature_extract()
                if not result.success:
                    print(f"WARNING: surfaceFeatureExtract failed, continuing...")
                
                result = self.run_snappy()
                if not result.success:
                    print(f"ERROR: snappyHexMesh failed")
                    return False
            else:
                if self.verbose:
                    print("  No STL files found, skipping snappyHexMesh")
        
        # Check mesh
        result = self.run_check_mesh()
        if not result.success:
            print(f"WARNING: checkMesh reported issues")
        
        # Solve
        result = self.run_solver(solver)
        if not result.success:
            print(f"ERROR: {solver} failed")
            return False
        
        # Write cell centres for comparison
        result = self.run_write_cell_centres()
        if not result.success:
            print(f"WARNING: writeCellCentres failed (comparison may not work)")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("Workflow completed successfully!")
            print(f"{'='*60}\n")
        
        return True
    
    def clean(self):
        """Clean case using foamlib."""
        if self.verbose:
            print(f"Cleaning case: {self.case_dir}")
        self._foam_case.clean()
    
    def get_latest_time(self) -> Optional[float]:
        """Get the latest time value using foamlib."""
        times = list(self._foam_case)
        if times:
            return times[-1].time
        return None
    
    def get_velocity_field(self, time_idx: int = -1):
        """
        Get velocity field from a time directory.
        
        Args:
            time_idx: Time index (-1 for latest)
        
        Returns:
            numpy array of velocity field (N, 3)
        """
        import numpy as np
        time_dir = self._foam_case[time_idx]
        U = time_dir["U"]
        return np.array(U.internal_field)
    
    def get_pressure_field(self, time_idx: int = -1):
        """
        Get pressure field from a time directory.
        
        Args:
            time_idx: Time index (-1 for latest)
        
        Returns:
            numpy array of pressure field (N,)
        """
        import numpy as np
        time_dir = self._foam_case[time_idx]
        p = time_dir["p"]
        p_internal = p.internal_field
        # Handle uniform field (single value)
        if isinstance(p_internal, (int, float)):
            return np.full(len(self.get_velocity_field(time_idx)), p_internal)
        return np.array(p_internal)
    
    def get_cell_centres(self, time_idx: int = -1):
        """
        Get cell centre coordinates.
        
        Must run run_write_cell_centres() first, or the C field must exist.
        
        Returns:
            numpy array (N, 3) of cell centre coordinates
        """
        import numpy as np
        
        # Check if C exists in time directory
        time_dir = self._foam_case[time_idx]
        
        try:
            C = time_dir["C"]
            return np.array(C.internal_field)
        except (KeyError, FileNotFoundError):
            # Need to generate cell centres
            print("Cell centres not found, running writeCellCentres...")
            result = self.run_write_cell_centres()
            if not result.success:
                raise RuntimeError("Failed to write cell centres")
            
            # Re-read (foamlib may need to refresh)
            self._foam_case = FoamCase(self.case_dir)
            time_dir = self._foam_case[time_idx]
            C = time_dir["C"]
            return np.array(C.internal_field)
    
    def get_fields_on_grid(
        self,
        x_range: tuple,
        y_range: tuple,
        resolution: tuple,
        z_plane: float = 0.05,
        time_idx: int = -1
    ):
        """
        Interpolate fields to a 2D structured grid for comparison.
        
        Args:
            x_range: (x_min, x_max)
            y_range: (y_min, y_max)
            resolution: (nx, ny)
            z_plane: Z coordinate to extract (midplane)
            time_idx: Time index (-1 for latest)
        
        Returns:
            dict with XX, YY, Vx, Vy, V_mag, p
        """
        from scipy.interpolate import griddata
        import numpy as np
        
        # Get cell centers
        C = self.get_cell_centres(time_idx)
        
        # Get fields
        U = self.get_velocity_field(time_idx)
        p = self.get_pressure_field(time_idx)
        
        # Filter cells near z_plane
        z_tol = 0.02  # Tolerance for z-plane extraction
        mask = np.abs(C[:, 2] - z_plane) < z_tol
        
        if mask.sum() < 10:
            print(f"Warning: Only {mask.sum()} cells near z={z_plane}, using all cells")
            mask = np.ones(len(C), dtype=bool)
        
        points_2d = C[mask, :2]
        U_filtered = U[mask]
        p_filtered = p[mask]
        
        # Create output grid
        nx, ny = resolution
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        XX, YY = np.meshgrid(x, y)
        
        # Interpolate
        Vx = griddata(points_2d, U_filtered[:, 0], (XX, YY), method='linear')
        Vy = griddata(points_2d, U_filtered[:, 1], (XX, YY), method='linear')
        p_grid = griddata(points_2d, p_filtered, (XX, YY), method='linear')
        V_mag = np.sqrt(Vx**2 + Vy**2)
        
        return {
            'XX': XX,
            'YY': YY,
            'Vx': Vx,
            'Vy': Vy,
            'V_mag': V_mag,
            'p': p_grid
        }
    
    def summary(self) -> str:
        """Get summary of all run results."""
        lines = [f"OpenFOAM Run Summary: {self.case_dir.name}"]
        lines.append("=" * 50)
        
        total_time = sum(r.runtime for r in self.results)
        successes = sum(1 for r in self.results if r.success)
        
        for result in self.results:
            lines.append(str(result))
        
        lines.append("-" * 50)
        lines.append(f"Total: {successes}/{len(self.results)} succeeded, {total_time:.1f}s")
        
        return "\n".join(lines)
