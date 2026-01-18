"""
YAML case file loader with validation.
"""

from pathlib import Path
from typing import Dict, Any
import yaml
import numpy as np

from ..config.schemas import SimulationConfig
from ..geometry.component import Component, Transform
from ..geometry.scene import Scene
from .geometry_io import GeometryReader


class CaseLoader:
    """Load and validate simulation cases from YAML files."""
    
    @staticmethod
    def load(filepath: str | Path) -> tuple[Scene, SimulationConfig]:
        """
        Load case file and create Scene.
        
        Args:
            filepath: Path to YAML case file
        
        Returns:
            Tuple of (Scene object, validated config)
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
        scene = CaseLoader._build_scene(config, base_path=filepath.parent)
        
        return scene, config
    
    @staticmethod
    def _build_scene(config: SimulationConfig, base_path: Path) -> Scene:
        """
        Build Scene from validated config.
        
        Args:
            config: Validated simulation config
            base_path: Base directory for resolving relative paths
        
        Returns:
            Scene object
        """
        components = []
        
        for comp_config in config.components:
            # Load geometry
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
            
            # Create component
            component = Component(
                name=comp_config.name,
                local_mesh=local_mesh,
                transform=transform,
                bc_type=bc_type,
                bc_value=bc_value,
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


def create_example_case(output_path: str | Path) -> None:
    """
    Create an example YAML case file.
    
    Args:
        output_path: Where to write the example file
    """
    example = {
        "name": "Example Two Squares",
        "case_type": "hardcoded_panels_2d",
        "description": "Example case with two squares in freestream",
        
        "freestream": {
            "velocity": [1.0, 0.0, 0.0]
        },
        
        "components": [
            {
                "name": "square_left",
                "geometry_file": "data/geometries/square_unit.json",
                "transform": {
                    "translation": [-2.0, 0.0, 0.0],
                    "rotation_deg": 0.0
                },
                "boundary_condition": {
                    "type": "wall"
                }
            },
            {
                "name": "square_right",
                "geometry_file": "data/geometries/square_unit.json",
                "transform": {
                    "translation": [2.0, 0.0, 0.0],
                    "rotation_deg": 45.0
                },
                "boundary_condition": {
                    "type": "wall"
                }
            }
        ],
        
        "solver": {
            "type": "constant_source",
            "tolerance": 1.0e-10
        },
        
        "output": {
            "directory": "./results/two_squares",
            "formats": ["vtk", "csv"]
        },
        
        "visualization": {
            "enabled": True,
            "show_mesh": True,
            "show_normals": True,
            "contour_resolution": [100, 100]
        }
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(example, f, default_flow_style=False, sort_keys=False)
