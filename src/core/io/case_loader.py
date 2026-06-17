"""
YAML case file loader with validation.
"""

from pathlib import Path
from typing import Dict, Any, Union
import yaml
import numpy as np

from ..config.schemas import SimulationConfig
from ..geometry.component import Component, Transform
from ..geometry.scene import Scene
from ..geometry.factory import GeometryFactory
from .geometry_io import GeometryReader
from .case import Case


class CaseLoader:
    """Load and validate simulation cases from YAML files."""
    
    @staticmethod
    def load(filepath: str | Path, mesh_level_index: int = 0) -> tuple[Scene, SimulationConfig]:
        """
        Load case file and create Scene.
        
        Args:
            filepath: Path to YAML case file
            mesh_level_index: Index into mesh_levels for parametric geometry (default=0)
        
        Returns:
            Tuple of (Scene object, validated config)
        
        Note:
            Consider using load_case() instead for cleaner access.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Case file not found: {filepath}")
        
        # Load YAML
        with open(filepath, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        # Validate with Pydantic
        config = SimulationConfig(**raw_config)
        
        # Build Scene from config
        scene = CaseLoader._build_scene(config, base_path=filepath.parent, mesh_level_index=mesh_level_index)
        
        return scene, config
    
    @staticmethod
    def _build_scene(config: SimulationConfig, base_path: Path, mesh_level_index: int = 0) -> Scene:
        """
        Build Scene from validated config.
        
        Args:
            config: Validated simulation config
            base_path: Base directory for resolving relative paths
            mesh_level_index: Index into per-component mesh_levels for parametric geometry (default=0)
        
        Returns:
            Scene object
        """
        components = []
        
        for comp_config in config.components:
            # Load geometry: parametric or file-based
            if comp_config.geometry is not None:
                resolution = None
                if comp_config.mesh_levels is not None and len(comp_config.mesh_levels) > 0:
                    num_levels = len(comp_config.mesh_levels)
                    level_idx = mesh_level_index
                    if level_idx < 0:
                        level_idx = num_levels + level_idx
                    if level_idx < 0 or level_idx >= num_levels:
                        raise IndexError(
                            f"Component '{comp_config.name}': mesh_level_index {mesh_level_index} out of range. "
                            f"Available levels: 0 to {num_levels - 1}"
                        )
                    resolution = comp_config.mesh_levels[level_idx]

                if comp_config.geometry.type == "external":
                    if comp_config.geometry.file is None:
                        raise ValueError(f"Component '{comp_config.name}': external geometry requires a 'file' parameter")
                    if comp_config.geometry.file.lower().endswith((".step", ".stl")):
                        from ..geometry.io.gmsh_reader import read_with_gmsh
                        geom_path = base_path / comp_config.geometry.file
                        scale = comp_config.geometry.parameters.get("scale", 1.0)
                        local_mesh = read_with_gmsh(geom_path, scale=scale, resolution=resolution)
                    else:
                        from ..geometry.io.stl_reader import read_mesh
                        geom_path = base_path / comp_config.geometry.file
                        local_mesh = read_mesh(geom_path)
                else:
                    # Parametric geometry - get resolution from component's mesh_levels
                    if resolution is None:
                        raise ValueError(
                            f"Component '{comp_config.name}' uses parametric geometry but "
                            f"no mesh_levels defined"
                        )
                    
                    geom_def = {
                        "type": comp_config.geometry.type,
                        "parameters": comp_config.geometry.parameters
                    }
                    local_mesh = GeometryFactory.create(geom_def, resolution)
            else:
                # Legacy file-based geometry
                if comp_config.geometry.file is None:
                    raise ValueError(f"Component '{comp_config.name}': must specify geometry or geometry_file")
                geom_path = base_path / comp_config.geometry_file
                local_mesh = GeometryReader.read(geom_path)
            
            # Build transform
            trans_config = comp_config.transform
            
            if trans_config.rotation_xyz_deg is not None:
                # 3D rotation
                rx, ry, rz = trans_config.rotation_xyz_deg
                transform = Transform.from_3d(
                    tx=trans_config.translation[0],
                    ty=trans_config.translation[1],
                    tz=trans_config.translation[2],
                    rx_deg=rx,
                    ry_deg=ry,
                    rz_deg=rz
                )
            else:
                # 2D rotation (about z-axis)
                transform = Transform.from_2d(
                    tx=trans_config.translation[0],
                    ty=trans_config.translation[1],
                    angle_deg=trans_config.rotation_deg
                )
            
            # Extract BC info
            bc_data = comp_config.boundary_condition
            bc_type = bc_data.get("type", "wall")
            bc_value = bc_data.get("value", None)
            bc_heat_flux = bc_data.get("heat_flux", None)
            
            # local_mesh.normals = -local_mesh.normals

            # Create component
            component = Component(
                name=comp_config.name,
                local_mesh=local_mesh,
                transform=transform,
                bc_type=bc_type,
                bc_value=bc_value,
                bc_heat_flux=bc_heat_flux,
                metadata={}
            )
            
            components.append(component)
        
        # Get freestream velocity
        freestream_vel = config.get_freestream_velocity()
        freestream = np.array(freestream_vel, dtype=np.float64)
        
        # Create scene
        scene = Scene(
            name=config.name,
            components=components,
            freestream=freestream,
            description=config.description
        )
        
        return scene
    
    @staticmethod
    def validate(filepath: str | Path) -> bool:
        """
        Validate case file without building scene.
        
        Args:
            filepath: Path to YAML case file
        
        Returns:
            True if valid, raises ValidationError otherwise
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        # This will raise ValidationError if invalid
        SimulationConfig(**raw_config)
        
        return True

    @staticmethod
    def load_case(case_dir: str | Path, mesh_level_index: int = 0) -> Case:
        """
        Load a case directory and return a Case object.
        
        This is the recommended way to load cases. Provides clean access:
            case = CaseLoader.load_case('cases/cylinder_flow')
            print(case.name)
            print(case.x_range, case.y_range)
            mesh = case.mesh
        
        For parametric cases, specify mesh level:
            case = CaseLoader.load_case('cases/single_square', mesh_level_index=2)
            # or use finest level:
            case = CaseLoader.load_case('cases/single_square', mesh_level_index=-1)
        
        Note:
            Each component with parametric geometry has its own mesh_levels.
            The mesh_level_index applies to all parametric components in the case.
        
        Args:
            case_dir: Path to case directory (containing case.yaml)
            mesh_level_index: Index into per-component mesh_levels for parametric geometry (default=0, use -1 for finest)
        
        Returns:
            Case object with scene, config, and helper properties
        """
        case_dir = Path(case_dir)
        case_file = case_dir / "case.yaml"
        
        if not case_file.exists():
            raise FileNotFoundError(f"No case.yaml found in {case_dir}")
        
        # Handle negative indexing for mesh levels (e.g., -1 for finest)
        # Note: For simplicity, we assume all components have the same number of levels
        # If they differ, the first parametric component determines the index conversion
        if mesh_level_index < 0:
            # Load config to get number of levels from first parametric component
            with open(case_file, 'r') as f:
                raw_config = yaml.safe_load(f)
            config = SimulationConfig(**raw_config)
            
            # Find first parametric component
            for comp in config.components:
                if comp.mesh_levels is not None:
                    mesh_level_index = len(comp.mesh_levels) + mesh_level_index
                    break
            else:
                # No parametric components, default to 0
                mesh_level_index = 0
        
        scene, config = CaseLoader.load(case_file, mesh_level_index=mesh_level_index)
        
        return Case(
            scene=scene,
            config=config,
            case_dir=case_dir,
            mesh_level_index=mesh_level_index
        )
