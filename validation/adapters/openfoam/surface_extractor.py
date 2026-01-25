"""
OpenFOAM surface data extractor.

Extracts surface quantities (velocity, pressure) from OpenFOAM wall patches
for comparison with panel method results.
"""

from pathlib import Path
from typing import Optional, List
import numpy as np
from numpy.typing import NDArray

try:
    from foamlib import FoamCase
except ImportError:
    raise ImportError(
        "foamlib is required for OpenFOAM surface extraction. "
        "Install with: pip install foamlib"
    )

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from postprocessing.surface import SurfaceData


class OpenFOAMSurfaceExtractor:
    """
    Extract surface data from OpenFOAM wall patches.
    
    Reads wall boundary field data (velocity, pressure) from OpenFOAM results
    and converts to SurfaceData format compatible with panel method output.
    
    Attributes:
        case_path: Path to OpenFOAM case directory
        foam_case: foamlib FoamCase object
        time_idx: Time index to extract (-1 for latest)
    """
    
    def __init__(
        self,
        case_path: Path | str,
        time_idx: int = -1
    ):
        """
        Initialize surface extractor.
        
        Args:
            case_path: Path to OpenFOAM case directory
            time_idx: Time index to extract (-1 for latest)
        """
        self.case_path = Path(case_path)
        self.time_idx = time_idx
        
        if not self.case_path.exists():
            raise FileNotFoundError(f"OpenFOAM case not found: {case_path}")
        
        # Load case with foamlib
        self.foam_case = FoamCase(self.case_path)
        
        # Get time directory
        times = list(self.foam_case)
        if not times:
            raise RuntimeError(f"No time directories found in {case_path}")
        
        self._time_dir = times[time_idx]
    
    def extract_patch(
        self,
        patch_name: str,
        reference_pressure: float = 0.0,
        density: float = 1.0,
        v_inf: float = 1.0
    ) -> SurfaceData:
        """
        Extract surface data from a single wall patch.
        
        Args:
            patch_name: Name of wall patch
            reference_pressure: Reference pressure for Cp calculation
            density: Fluid density
            v_inf: Freestream velocity magnitude
        
        Returns:
            SurfaceData object with x, y, s, Vt, Cp
        """
        # Read velocity and pressure boundary fields
        U_field = self._time_dir["U"]
        p_field = self._time_dir["p"]
        
        try:
            U_boundary = U_field.boundary_field[patch_name]
            p_boundary = p_field.boundary_field[patch_name]
        except KeyError:
            available = list(U_field.boundary_field.keys())
            raise KeyError(
                f"Patch '{patch_name}' not found. "
                f"Available patches: {available}"
            )
        
        # Direct patch extraction not yet implemented
        # Use sample_at_points() instead to sample at panel centers
        raise NotImplementedError(
            "Direct patch extraction not yet implemented. "
            "Use sample_at_points() to sample OpenFOAM fields at panel centers."
        )
    
    def extract_all_walls(
        self,
        reference_pressure: float = 0.0,
        density: float = 1.0,
        v_inf: float = 1.0,
        exclude_patches: Optional[List[str]] = None
    ) -> SurfaceData:
        """
        Extract surface data from all wall patches.
        
        Args:
            reference_pressure: Reference pressure for Cp calculation
            density: Fluid density
            v_inf: Freestream velocity magnitude
            exclude_patches: List of patch names to exclude (e.g., ['inlet', 'outlet'])
        
        Returns:
            Combined SurfaceData with all wall patches
        """
        if exclude_patches is None:
            exclude_patches = ['inlet', 'outlet', 'top', 'bottom', 'frontAndBack']
        
        # Get all patches
        U_field = self._time_dir["U"]
        all_patches = list(U_field.boundary_field.keys())
        
        # Filter to wall patches only
        wall_patches = [
            p for p in all_patches
            if p not in exclude_patches
        ]
        
        if not wall_patches:
            raise RuntimeError(
                f"No wall patches found. Available patches: {all_patches}"
            )
        
        # Extract each patch
        surface_data_list = []
        for i, patch_name in enumerate(wall_patches):
            data = self.extract_patch(
                patch_name,
                reference_pressure=reference_pressure,
                density=density,
                v_inf=v_inf
            )
            # Assign component ID
            data.component_id[:] = i
            surface_data_list.append(data)
        
        # Concatenate all patches
        return self._concatenate_surface_data(surface_data_list)
    
    def sample_at_points(
        self,
        points: NDArray[np.float64],
        reference_pressure: float = 0.0,
        density: float = 1.0,
        v_inf: float = 1.0
    ) -> SurfaceData:
        """
        Sample OpenFOAM fields at specified surface points.
        
        This is the recommended approach for validation: sample OpenFOAM
        at the same panel center locations used by the panel method.
        
        Uses scipy interpolation to sample the internal field at arbitrary points.
        
        Args:
            points: Surface points to sample at (N, 3) or (N, 2)
            reference_pressure: Reference pressure for Cp calculation
            density: Fluid density
            v_inf: Freestream velocity magnitude
        
        Returns:
            SurfaceData with interpolated values at specified points
        """
        from scipy.interpolate import griddata
        
        # Ensure points are (N, 3)
        if points.shape[1] == 2:
            points_3d = np.column_stack([points, np.zeros(len(points))])
        else:
            points_3d = points
        
        # Get cell centres and fields
        U_field = self._time_dir["U"]
        p_field = self._time_dir["p"]
        
        # Read internal field (volume)
        U_internal = np.array(U_field.internal_field)
        p_internal = np.array(p_field.internal_field)
        
        # Handle uniform pressure field
        if np.isscalar(p_internal) or p_internal.ndim == 0:
            p_internal = np.full(len(U_internal), float(p_internal))
        
        # Get cell centres (need to run writeCellCentres first)
        # This will be the interpolation source points
        try:
            C_field = self._time_dir["C"]
            cell_centres = np.array(C_field.internal_field)
        except (KeyError, FileNotFoundError):
            raise RuntimeError(
                "Cell centres not found. Run 'writeCellCentres' "
                "in your OpenFOAM case first."
            )
        
        # Interpolate velocity to surface points
        # For 2D cases, only use x,y coordinates
        Vx_interp = griddata(
            cell_centres[:, :2], U_internal[:, 0], 
            points_3d[:, :2], method='linear', fill_value=v_inf
        )
        Vy_interp = griddata(
            cell_centres[:, :2], U_internal[:, 1],
            points_3d[:, :2], method='linear', fill_value=0.0
        )
        
        # Tangential velocity magnitude
        Vt = np.sqrt(Vx_interp**2 + Vy_interp**2)
        
        # Interpolate pressure
        p_interp = griddata(
            cell_centres[:, :2], p_internal,
            points_3d[:, :2], method='linear', fill_value=reference_pressure
        )
        
        # Compute pressure coefficient
        q_inf = 0.5 * density * v_inf**2
        Cp = (p_interp - reference_pressure) / q_inf
        
        # Extract x, y coordinates
        x = points_3d[:, 0]
        y = points_3d[:, 1]
        
        # Compute arc length
        s = self._compute_arc_length(x, y)
        
        # Component ID (not meaningful for point sampling)
        component_id = np.zeros(len(x), dtype=np.int32)
        
        # Normal velocity (should be ~0 for wall boundaries in potentialFoam)
        # For now, set to None since we're sampling in the volume, not on the wall
        Vn = np.zeros(len(x), dtype=np.float64)  # Wall boundary: Vn = 0
        
        return SurfaceData(
            x=x,
            y=y,
            s=s,
            Vt=Vt,
            Vn=Vn,
            Cp=Cp,
            component_id=component_id
        )
    
    @staticmethod
    def _compute_arc_length(x: NDArray, y: NDArray) -> NDArray:
        """
        Compute cumulative arc length along curve.
        
        Args:
            x: X coordinates
            y: Y coordinates
        
        Returns:
            Cumulative arc length (starts at 0)
        """
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate([[0], np.cumsum(ds)])
        return s
    
    @staticmethod
    def _concatenate_surface_data(data_list: List[SurfaceData]) -> SurfaceData:
        """
        Concatenate multiple SurfaceData objects.
        
        Arc length is reset for each component (starts at 0).
        
        Args:
            data_list: List of SurfaceData objects
        
        Returns:
            Combined SurfaceData
        """
        x = np.concatenate([d.x for d in data_list])
        y = np.concatenate([d.y for d in data_list])
        s = np.concatenate([d.s for d in data_list])
        Vt = np.concatenate([d.Vt for d in data_list])
        Cp = np.concatenate([d.Cp for d in data_list])
        component_id = np.concatenate([d.component_id for d in data_list])
        
        return SurfaceData(
            x=x,
            y=y,
            s=s,
            Vt=Vt,
            Cp=Cp,
            component_id=component_id
        )
