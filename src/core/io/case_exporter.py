"""
Case exporter - Convert programmatic geometry to case folder structure.

Allows you to:
1. Generate geometry in code
2. Export to standard case folder (case.yaml + shapes/*.json)
3. Run with standard demo scripts later

Usage:
    from core.io.case_exporter import CaseExporter
    
    # From Scene
    exporter = CaseExporter.from_scene(scene, solver_type='constant_source')
    exporter.export('cases/my_new_case')
    
    # From single Mesh
    exporter = CaseExporter.from_mesh(mesh, name='single_body', freestream=[1,0,0])
    exporter.export('cases/my_mesh_case')
"""

from pathlib import Path
from typing import Optional, List, Tuple, Union
import json
import yaml
import numpy as np
from numpy.typing import NDArray

from ..geometry import Mesh, Scene, Component, Transform


class CaseExporter:
    """Export programmatic geometry to case folder structure."""
    
    def __init__(self,
                 name: str,
                 description: str = "",
                 freestream: Tuple[float, float, float] = (1.0, 0.0, 0.0),
                 solver_type: str = "constant_source",
                 tolerance: float = 1e-10):
        """
        Initialize exporter with case metadata.
        
        Args:
            name: Case name
            description: Case description
            freestream: Freestream velocity (Vx, Vy, Vz)
            solver_type: Solver type string
            tolerance: Solver tolerance
        """
        self.name = name
        self.description = description
        self.freestream = list(freestream)
        self.solver_type = solver_type
        self.tolerance = tolerance
        
        self.components: List[dict] = []
        self.meshes: List[Tuple[str, Mesh]] = []  # (filename, mesh) pairs
        
        # Visualization defaults
        self.viz_domain = {
            'x_range': [-3.0, 5.0],
            'y_range': [-3.0, 3.0]
        }
        self.viz_resolution = [150, 120]
        
        # Fluid properties (optional, uses defaults if not set)
        self.fluid: Optional[dict] = None
    
    @classmethod
    def from_scene(cls,
                   scene: Scene,
                   solver_type: str = "constant_source",
                   tolerance: float = 1e-10) -> 'CaseExporter':
        """
        Create exporter from an existing Scene.
        
        Args:
            scene: Scene object with components
            solver_type: Solver type
            tolerance: Solver tolerance
        
        Returns:
            CaseExporter ready to export
        """
        exporter = cls(
            name=scene.name,
            description=scene.description,
            freestream=tuple(scene.freestream.tolist()),
            solver_type=solver_type,
            tolerance=tolerance
        )
        
        for comp in scene.components:
            exporter.add_component(comp)
        
        # Auto-compute visualization domain from geometry
        exporter._auto_domain(scene)
        
        return exporter
    
    @classmethod
    def from_mesh(cls,
                  mesh: Mesh,
                  name: str = "single_body",
                  description: str = "",
                  freestream: Tuple[float, float, float] = (1.0, 0.0, 0.0),
                  solver_type: str = "constant_source") -> 'CaseExporter':
        """
        Create exporter from a single Mesh.
        
        Args:
            mesh: Mesh object
            name: Case name
            description: Case description
            freestream: Freestream velocity
            solver_type: Solver type
        
        Returns:
            CaseExporter ready to export
        """
        exporter = cls(
            name=name,
            description=description,
            freestream=freestream,
            solver_type=solver_type
        )
        
        # Wrap mesh in a component
        component = Component(
            name="body",
            local_mesh=mesh,
            transform=Transform.identity(),
            bc_type="wall"
        )
        exporter.add_component(component)
        
        # Auto domain
        x_min, x_max = mesh.nodes[:, 0].min(), mesh.nodes[:, 0].max()
        y_min, y_max = mesh.nodes[:, 1].min(), mesh.nodes[:, 1].max()
        padding = max(x_max - x_min, y_max - y_min)
        exporter.viz_domain = {
            'x_range': [float(x_min - padding), float(x_max + 2*padding)],
            'y_range': [float(y_min - padding), float(y_max + padding)]
        }
        
        return exporter
    
    def add_component(self, component: Component):
        """
        Add a component to the case.
        
        Args:
            component: Component object
        """
        # Generate shape filename
        shape_filename = f"{component.name}.json"
        
        # Store mesh for later export
        self.meshes.append((shape_filename, component.local_mesh))
        
        # Extract transform
        trans = component.transform
        translation = trans.translation.tolist()
        
        # For 2D, extract rotation angle from matrix
        # rotation_deg = atan2(R[1,0], R[0,0]) in degrees
        rot_mat = trans.rotation_matrix
        angle_deg = float(np.degrees(np.arctan2(rot_mat[1, 0], rot_mat[0, 0])))
        
        # Build component config
        comp_config = {
            'name': component.name,
            'geometry_file': f"shapes/{shape_filename}",
            'transform': {
                'translation': [translation[0], translation[1], translation[2]],
                'rotation_deg': angle_deg
            },
            'boundary_condition': {
                'type': component.bc_type
            }
        }
        
        if component.bc_value is not None:
            comp_config['boundary_condition']['value'] = component.bc_value
        
        self.components.append(comp_config)
    
    def set_visualization_domain(self,
                                  x_range: Tuple[float, float],
                                  y_range: Tuple[float, float],
                                  resolution: Tuple[int, int] = (150, 120)):
        """Set visualization domain and resolution."""
        self.viz_domain = {
            'x_range': list(x_range),
            'y_range': list(y_range)
        }
        self.viz_resolution = list(resolution)
    
    def set_fluid(self,
                  density: float = 1.225,
                  viscosity: Optional[float] = None,
                  gravity: float = 0.0,
                  reference_pressure: float = 101325.0,
                  reference_type: str = "freestream"):
        """
        Set fluid properties for post-processing.
        
        Args:
            density: Fluid density [kg/m³]
            viscosity: Dynamic viscosity [Pa·s] (optional)
            gravity: Gravitational acceleration [m/s²]
            reference_pressure: Reference pressure [Pa]
            reference_type: "freestream" or "outlet"
        """
        self.fluid = {
            'density': density,
            'gravity': gravity,
            'reference_pressure': reference_pressure,
            'reference_type': reference_type
        }
        if viscosity is not None:
            self.fluid['viscosity'] = viscosity
    
    def set_fluid_from_state(self, fluid_state: 'FluidState'):
        """
        Set fluid properties from a FluidState object.
        
        Args:
            fluid_state: FluidState object from postprocessing.fluid
        """
        self.fluid = fluid_state.to_dict()
    
    def _auto_domain(self, scene: Scene):
        """Automatically compute visualization domain from scene."""
        mesh = scene.assemble()
        x_min, x_max = mesh.nodes[:, 0].min(), mesh.nodes[:, 0].max()
        y_min, y_max = mesh.nodes[:, 1].min(), mesh.nodes[:, 1].max()
        
        padding = max(x_max - x_min, y_max - y_min) * 0.5
        
        self.viz_domain = {
            'x_range': [float(x_min - padding), float(x_max + 2*padding)],
            'y_range': [float(y_min - padding), float(y_max + padding)]
        }
    
    def export(self, case_dir: Union[str, Path], overwrite: bool = False):
        """
        Export case to folder structure.
        
        Creates:
            case_dir/
                case.yaml
                shapes/
                    component1.json
                    component2.json
                    ...
        
        Args:
            case_dir: Target case directory
            overwrite: Allow overwriting existing case
        """
        case_dir = Path(case_dir)
        
        if case_dir.exists() and not overwrite:
            raise FileExistsError(
                f"Case directory already exists: {case_dir}\n"
                f"Use overwrite=True to replace."
            )
        
        # Create directories
        shapes_dir = case_dir / "shapes"
        shapes_dir.mkdir(parents=True, exist_ok=True)
        
        # Export geometry JSON files
        for filename, mesh in self.meshes:
            self._export_mesh_json(mesh, shapes_dir / filename)
        
        # Build case.yaml
        case_config = {
            'name': self.name,
            'case_type': 'hardcoded_panels_2d' if self.meshes[0][1].dimension == 2 else 'hardcoded_panels_3d',
            'description': self.description,
            'freestream': {
                'velocity': self.freestream
            },
            'components': self.components,
            'solver': {
                'type': self.solver_type,
                'tolerance': self.tolerance
            },
            'visualization': {
                'domain': self.viz_domain,
                'resolution': self.viz_resolution
            }
        }
        
        # Add fluid properties if set
        if self.fluid is not None:
            case_config['fluid'] = self.fluid
        
        # Write case.yaml
        case_file = case_dir / "case.yaml"
        with open(case_file, 'w') as f:
            yaml.dump(case_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Exported case to: {case_dir}")
        print(f"  - case.yaml")
        for filename, _ in self.meshes:
            print(f"  - shapes/{filename}")
    
    def _export_mesh_json(self, mesh: Mesh, filepath: Path):
        """Export a single mesh to JSON format."""
        nodes_list = mesh.nodes.tolist()
        panels_list = mesh.panels.tolist()
        
        data = {
            'format': f'panels_{mesh.dimension}d',
            'nodes': nodes_list,
            'panels': panels_list,
            'normal_direction': 'outward'
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
