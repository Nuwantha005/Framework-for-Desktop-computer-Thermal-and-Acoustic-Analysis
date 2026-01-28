"""
OpenFOAM case generator using foamlib (template-based approach).

Replaces the old f-string/regex-based generator with a clean foamlib implementation.
Uses template cloning and structured edits for robustness and maintainability.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
import shutil

from foamlib import FoamCase, Dimensioned, DimensionSet

from .geometry_converter import GeometryConverter
from .case_generator import MeshSettings  # Reuse existing dataclass


def sanitize_name(name: str) -> str:
    """
    Convert a name to a valid OpenFOAM identifier.
    
    Replaces spaces with underscores and removes special characters.
    """
    # Replace spaces with underscores
    safe_name = name.replace(" ", "_")
    # Remove or replace other problematic characters
    safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in safe_name)
    # Remove leading/trailing underscores
    safe_name = safe_name.strip("_")
    # Ensure it doesn't start with a number (OpenFOAM requirement)
    if safe_name and safe_name[0].isdigit():
        safe_name = "geo_" + safe_name
    return safe_name


class FoamlibCaseGenerator:
    """
    Generate OpenFOAM cases using foamlib template-based approach.
    
    Advantages over old generator:
    - No f-strings or regex - uses foamlib's structured API
    - Template ensures valid OpenFOAM syntax
    - Easier to maintain and extend
    - Ready for Phase 5 parametric geometry
    
    Usage:
        from core.io import CaseLoader
        from validation.adapters.openfoam import FoamlibCaseGenerator, MeshSettings
        
        case = CaseLoader.load_case("cases/single_square")
        
        generator = FoamlibCaseGenerator(
            case=case,
            output_dir="validation_results/single_square/openfoam",
            mesh_settings=MeshSettings(background_cells_per_unit=10.0)
        )
        
        of_case_dir = generator.generate()
    """
    
    # Template location (relative to this file)
    TEMPLATE_DIR = Path(__file__).parent.parent.parent.parent / "templates" / "openfoam" / "potentialFoam2D"
    
    def __init__(
        self,
        case,  # Panel method Case object
        output_dir: Path | str,
        solver_type: str = "potentialFoam",
        mesh_settings: Optional[MeshSettings] = None,
        domain_padding: float = 2.0,
        n_processors: int = 4,  # For parallel snappyHexMesh
    ):
        """
        Initialize foamlib-based case generator.
        
        Args:
            case: Panel method Case object from CaseLoader
            output_dir: Directory for generated OpenFOAM case
            solver_type: "potentialFoam" (others can be added later)
            mesh_settings: Mesh generation settings
            domain_padding: Extra domain space around bodies
            n_processors: Number of processors for parallel operations
        """
        self.case = case
        self.output_dir = Path(output_dir)
        self.solver_type = solver_type
        self.mesh_settings = mesh_settings or MeshSettings()
        self.domain_padding = domain_padding
        self.n_processors = n_processors
        
        # Verify template exists
        if not self.TEMPLATE_DIR.exists():
            raise FileNotFoundError(
                f"OpenFOAM template not found: {self.TEMPLATE_DIR}\n"
                f"Run: python scripts/create_openfoam_template.py"
            )
        
        # Compute domain bounds
        self.domain = self._compute_domain()
    
    def _compute_domain(self) -> dict:
        """Compute OpenFOAM domain bounds with padding."""
        x_min, x_max = self.case.x_range
        y_min, y_max = self.case.y_range
        
        # More padding downstream
        inlet_pad = self.domain_padding
        outlet_pad = self.domain_padding * 2
        lateral_pad = self.domain_padding
        
        return {
            'x': (x_min - inlet_pad, x_max + outlet_pad),
            'y': (y_min - lateral_pad, y_max + lateral_pad),
            'z': (0.0, self.mesh_settings.z_thickness),
        }
    
    def generate(self) -> Path:
        """
        Generate complete OpenFOAM case from template.
        
        Returns:
            Path to generated case directory
        """
        # Clone template
        template = FoamCase(self.TEMPLATE_DIR)
        
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        
        foam_case = template.copy(self.output_dir)
        
        # Generate geometry STL files
        self._generate_geometry(foam_case)
        
        # Configure case using foamlib
        self._set_boundary_conditions(foam_case)
        self._set_block_mesh(foam_case)
        self._set_snappy_hex_mesh(foam_case)
        self._set_control_dict(foam_case)
        self._set_transport_properties(foam_case)
        self._set_decompose_par(foam_case)
        self._set_surface_feature_extract(foam_case)
        self._set_function_objects(foam_case)
        
        return self.output_dir
    
    def _generate_geometry(self, foam_case: FoamCase):
        """Generate STL files from panel method geometry."""
        stl_dir = self.output_dir / "constant" / "triSurface"
        stl_dir.mkdir(parents=True, exist_ok=True)
        
        converter = GeometryConverter(extrusion_depth=self.mesh_settings.z_thickness)
        
        # Generate STL for the scene
        # Phase 5 will handle multi-component scenes with separate STL files
        stl_paths = converter.convert_scene(
            self.case.scene,
            output_dir=stl_dir,
            binary=False,
            combined=True  # Single combined STL for now
        )
        
        # Sanitize STL filenames to valid OpenFOAM identifiers
        # Rename files and track sanitized names
        self.component_names = []
        for path in stl_paths:
            safe_name = sanitize_name(path.stem)
            new_path = path.parent / f"{safe_name}.stl"
            if path != new_path:
                path.rename(new_path)
            self.component_names.append(safe_name)
    
    def _set_boundary_conditions(self, foam_case: FoamCase):
        """Set initial and boundary conditions using foamlib."""
        # Freestream velocity
        vx, vy, vz = self.case.freestream
        
        # 0/U
        with foam_case[0]["U"] as U:
            U.dimensions = DimensionSet(length=1, time=-1)
            U.internal_field = [vx, vy, vz]
            U.boundary_field["inlet"]["value"] = [vx, vy, vz]
            # Outlet, top, bottom, front, back remain from template
            
            # Add wall patches for bodies (if any)
            # For inviscid potential flow, use slip condition to allow tangential velocity
            for name in self.component_names:
                U.boundary_field[name] = {
                    "type": "slip"
                }
        
        # 0/p - keep template defaults
        with foam_case[0]["p"] as p:
            # Add wall patches
            for name in self.component_names:
                p.boundary_field[name] = {"type": "zeroGradient"}
    
    def _set_block_mesh(self, foam_case: FoamCase):
        """Configure blockMeshDict using foamlib."""
        (x0, x1) = self.domain['x']
        (y0, y1) = self.domain['y']
        (z0, z1) = self.domain['z']
        
        # Compute cell counts
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        
        nx = int(dx * self.mesh_settings.background_cells_per_unit)
        ny = int(dy * self.mesh_settings.background_cells_per_unit)
        nz = 1  # 2D case
        
        with foam_case.block_mesh_dict as f:
            f["scale"] = 1
            
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
            
            f["blocks"] = [
                "hex",
                [0, 1, 2, 3, 4, 5, 6, 7],
                [nx, ny, nz],
                "simpleGrading",
                [1, 1, 1],
            ]
            
            # Boundary patches remain from template
    
    def _set_snappy_hex_mesh(self, foam_case: FoamCase):
        """Configure snappyHexMeshDict using foamlib."""
        snappy = foam_case.file("system/snappyHexMeshDict")
        
        surface_level = self.mesh_settings.refinement_level
        feature_level = surface_level
        
        with snappy as f:
            # Geometry
            f["geometry"] = {
                name: {
                    "type": "triSurfaceMesh",
                    "file": f"{name}.stl",
                }
                for name in self.component_names
            }
            
            # Features
            f["castellatedMeshControls", "features"] = [
                {"file": f"{name}.eMesh", "level": feature_level}
                for name in self.component_names
            ]
            
            # Refinement surfaces
            f["castellatedMeshControls", "refinementSurfaces"] = {
                name: {
                    "level": [surface_level, surface_level],
                    "patchInfo": {"type": "wall"},
                }
                for name in self.component_names
            }
            
            # Location in mesh (inside flow domain, outside bodies)
            # Use a point near inlet
            x0 = self.domain['x'][0]
            y_mid = sum(self.domain['y']) / 2
            z_mid = sum(self.domain['z']) / 2
            f["castellatedMeshControls", "locationInMesh"] = [x0 + 0.01, y_mid, z_mid]
            
            # Scale max cells with refinement level
            scale_factor = 2 ** max(0, surface_level - 2)
            f["castellatedMeshControls", "maxLocalCells"] = int(100000 * scale_factor)
            f["castellatedMeshControls", "maxGlobalCells"] = int(2000000 * scale_factor)
    
    def _set_control_dict(self, foam_case: FoamCase):
        """Configure controlDict using foamlib."""
        with foam_case.control_dict as f:
            f["application"] = self.solver_type
            # Other settings keep template defaults
    
    def _set_transport_properties(self, foam_case: FoamCase):
        """Set transport properties using foamlib."""
        # Get viscosity from case config
        if hasattr(self.case, 'viscosity') and self.case.viscosity is not None:
            nu = self.case.viscosity
        else:
            nu = 1.5e-5  # Air at 20°C
        
        with foam_case.transport_properties as f:
            f["nu"] = Dimensioned(nu, DimensionSet(length=2, time=-1), "nu")
    
    def _set_decompose_par(self, foam_case: FoamCase):
        """Configure parallel decomposition using foamlib."""
        with foam_case.decompose_par_dict as f:
            f["numberOfSubdomains"] = self.n_processors
            f["method"] = "scotch"
    
    def _set_surface_feature_extract(self, foam_case: FoamCase):
        """Configure surfaceFeatureExtractDict using foamlib."""
        sfe = foam_case.file("system/surfaceFeatureExtractDict")
        
        with sfe as f:
            for name in self.component_names:
                f[f"{name}.stl"] = {
                    "extractionMethod": "extractFromSurface",
                    "includedAngle": 150,
                    "subsetFeatures": {
                        "nonManifoldEdges": "yes",
                        "openEdges": "yes",
                    },
                }
    
    def _set_function_objects(self, foam_case: FoamCase):
        """
        Create function objects for post-processing.
        
        Generates surfaceFieldValue function objects to extract wall velocities
        for validation against panel method results. Creates a separate function
        object for each component (wall patch).
        
        Function objects are written to system/functionObjects/ directory and
        included in controlDict.
        """
        func_obj_dir = self.output_dir / "system" / "functionObjects"
        func_obj_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a function object for each component (wall patch)
        for comp_name in self.component_names:
            func_obj_name = f"wallSlipVelocity_{comp_name}"
            func_obj_file = func_obj_dir / func_obj_name
            
            # Write function object definition
            # Use manual text generation since foamlib doesn't handle function objects well
            content = f"""{func_obj_name}
{{
    type            surfaceFieldValue;
    libs            (fieldFunctionObjects);
    enabled         true;

    regionType      patch;
    name            {comp_name};

    operation       none;
    fields          (U p);

    executeControl  onEnd;
    writeControl    onEnd;
        
    writeFields     true;
    surfaceFormat   vtk;
}}
"""
            func_obj_file.write_text(content)
        
        # Update controlDict to include function objects
        # Note: foamlib's controlDict access doesn't support #include directives well,
        # so we manually append to the file
        control_dict_path = self.output_dir / "system" / "controlDict"
        
        with open(control_dict_path, 'r') as f:
            content = f.read()
        
        # Check if functions block already exists
        if 'functions' not in content:
            # Add functions block before the closing of FoamFile
            # Find where to insert (before final closing brace or at end)
            if content.rstrip().endswith('}'):
                # Remove trailing whitespace and closing brace
                content = content.rstrip()[:-1].rstrip()
            
            # Add functions block
            functions_block = "\n\nfunctions\n{\n"
            for comp_name in self.component_names:
                func_obj_name = f"wallSlipVelocity_{comp_name}"
                functions_block += f'    #include "functionObjects/{func_obj_name}"\n'
            functions_block += "}\n"
            
            content += functions_block
            
            with open(control_dict_path, 'w') as f:
                f.write(content)


# Backward compatibility: keep old name as alias
OpenFOAMCaseGenerator = FoamlibCaseGenerator
