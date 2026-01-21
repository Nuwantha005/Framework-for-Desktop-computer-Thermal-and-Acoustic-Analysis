"""
OpenFOAM case generator.

Generates complete OpenFOAM case structure from panel method case files.
Uses blockMesh for background mesh and snappyHexMesh for body refinement.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Literal
import shutil
import numpy as np

from .geometry_converter import GeometryConverter


@dataclass
class MeshSettings:
    """Settings for OpenFOAM mesh generation."""
    # Background mesh (blockMesh)
    background_cells_per_unit: float = 10.0  # Cells per unit length
    
    # Boundary layer (optional, for simpleFoam)
    n_boundary_layers: int = 0
    boundary_layer_thickness: float = 0.1
    boundary_layer_expansion: float = 1.2
    
    # Refinement near bodies
    refinement_level: int = 2
    
    # Z extrusion for 2D
    z_thickness: float = 0.1


@dataclass 
class SolverType:
    """OpenFOAM solver configuration."""
    name: Literal["potentialFoam", "simpleFoam", "chtMultiRegionSimpleFoam"]
    
    # For simpleFoam
    turbulence_model: Optional[str] = None  # "laminar", "kEpsilon", "kOmegaSST"
    
    # For chtMultiRegionSimpleFoam
    thermal: bool = False


class OpenFOAMCaseGenerator:
    """
    Generate OpenFOAM case from panel method Case object.
    
    Creates:
    - 0/ directory with initial/boundary conditions
    - constant/ directory with transport properties and geometry STL
    - system/ directory with mesh and solver settings
    
    Usage:
        from core.io import CaseLoader
        from validation import OpenFOAMCaseGenerator
        
        case = CaseLoader.load_case("cases/two_rounded_rects")
        generator = OpenFOAMCaseGenerator(
            case=case,
            output_dir="validation_results/two_rounded_rects",
            solver_type="potentialFoam"
        )
        of_case_path = generator.generate()
    """
    
    def __init__(
        self,
        case,  # Case object from core.io
        output_dir: Path | str,
        solver_type: str = "potentialFoam",
        mesh_settings: Optional[MeshSettings] = None,
        domain_padding: float = 2.0,  # Extra space around bodies
    ):
        """
        Initialize case generator.
        
        Args:
            case: Panel method Case object
            output_dir: Directory to create OpenFOAM case in
            solver_type: "potentialFoam", "simpleFoam", or "chtMultiRegionSimpleFoam"
            mesh_settings: Mesh generation settings
            domain_padding: Extra domain size beyond viz bounds (for inlet/outlet)
        """
        self.case = case
        self.output_dir = Path(output_dir)
        self.solver_type = solver_type
        self.mesh_settings = mesh_settings or MeshSettings()
        self.domain_padding = domain_padding
        
        # Extract from case
        self.freestream = case.freestream
        self.v_inf = case.v_inf
        self.x_range = case.x_range
        self.y_range = case.y_range
        
        # Add padding to domain for proper inlet/outlet
        self.domain = self._compute_domain()
        
    def _compute_domain(self) -> Dict[str, Tuple[float, float]]:
        """Compute OpenFOAM domain bounds with padding."""
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        
        # Add padding - more on outlet side
        inlet_pad = self.domain_padding
        outlet_pad = self.domain_padding * 2  # More space downstream
        lateral_pad = self.domain_padding
        
        return {
            'x': (x_min - inlet_pad, x_max + outlet_pad),
            'y': (y_min - lateral_pad, y_max + lateral_pad),
            'z': (0.0, self.mesh_settings.z_thickness),
        }
    
    def generate(self) -> Path:
        """
        Generate complete OpenFOAM case.
        
        Returns:
            Path to generated case directory
        """
        # Create directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "0").mkdir(exist_ok=True)
        (self.output_dir / "constant").mkdir(exist_ok=True)
        (self.output_dir / "constant" / "triSurface").mkdir(exist_ok=True)
        (self.output_dir / "system").mkdir(exist_ok=True)
        
        # Generate components
        self._generate_geometry()
        self._generate_blockmesh_dict()
        self._generate_snappy_hex_mesh_dict()
        self._generate_control_dict()
        self._generate_fv_schemes()
        self._generate_fv_solution()
        self._generate_boundary_conditions()
        self._generate_transport_properties()
        self._generate_decompose_par_dict()
        self._generate_sample_dict()
        self._generate_run_scripts()
        
        return self.output_dir
    
    def _generate_geometry(self):
        """Convert and export geometry STL files."""
        converter = GeometryConverter(extrusion_depth=self.mesh_settings.z_thickness)
        
        # Export individual components (for snappyHexMesh refinement regions)
        stl_dir = self.output_dir / "constant" / "triSurface"
        
        for component in self.case.scene.components:
            converter.convert_component(component, stl_dir, binary=False)
    
    def _generate_blockmesh_dict(self):
        """Generate blockMeshDict for background mesh."""
        domain = self.domain
        z_thick = self.mesh_settings.z_thickness
        cells_per_unit = self.mesh_settings.background_cells_per_unit
        
        # Calculate cell counts
        x_len = domain['x'][1] - domain['x'][0]
        y_len = domain['y'][1] - domain['y'][0]
        
        nx = max(10, int(x_len * cells_per_unit))
        ny = max(10, int(y_len * cells_per_unit))
        nz = 1  # Single cell in z for 2D
        
        # Vertices
        x0, x1 = domain['x']
        y0, y1 = domain['y']
        z0, z1 = 0.0, z_thick
        
        content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
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
    class       dictionary;
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

scale   1;

vertices
(
    ({x0} {y0} {z0})  // 0
    ({x1} {y0} {z0})  // 1
    ({x1} {y1} {z0})  // 2
    ({x0} {y1} {z0})  // 3
    ({x0} {y0} {z1})  // 4
    ({x1} {y0} {z1})  // 5
    ({x1} {y1} {z1})  // 6
    ({x0} {y1} {z1})  // 7
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
            (0 4 7 3)
        );
    }}
    outlet
    {{
        type patch;
        faces
        (
            (1 2 6 5)
        );
    }}
    top
    {{
        type patch;
        faces
        (
            (3 7 6 2)
        );
    }}
    bottom
    {{
        type patch;
        faces
        (
            (0 1 5 4)
        );
    }}
    front
    {{
        type symmetry;
        faces
        (
            (0 3 2 1)
        );
    }}
    back
    {{
        type symmetry;
        faces
        (
            (4 5 6 7)
        );
    }}
);

// ************************************************************************* //
'''
        
        with open(self.output_dir / "system" / "blockMeshDict", 'w') as f:
            f.write(content)
    
    def _generate_snappy_hex_mesh_dict(self):
        """Generate snappyHexMeshDict for body refinement."""
        # Build geometry entries for all components
        geometry_entries = []
        refinement_entries = []
        feature_entries = []
        
        for comp in self.case.scene.components:
            stl_name = f"{comp.name}.stl"
            geometry_entries.append(f'''
        {comp.name}
        {{
            type triSurfaceMesh;
            file "{stl_name}";
        }}''')
            
            refinement_entries.append(f'''
            {comp.name}
            {{
                level ({self.mesh_settings.refinement_level} {self.mesh_settings.refinement_level});
                patchInfo
                {{
                    type wall;
                }}
            }}''')
            
            # Feature edge entries for better snapping
            feature_entries.append(f'''
        {{
            file "{comp.name}.eMesh";
            level {self.mesh_settings.refinement_level};
        }}''')
        
        geometry_str = "\n".join(geometry_entries)
        refinement_str = "\n".join(refinement_entries)
        features_str = "\n".join(feature_entries)
        
        # Location in mesh (point that must be in the fluid region)
        # Use a point far from bodies
        x_mid = sum(self.domain['x']) / 2
        y_mid = sum(self.domain['y']) / 2
        z_mid = self.mesh_settings.z_thickness / 2
        
        # Find a point definitely outside all bodies (near inlet)
        location_x = self.domain['x'][0] + 0.1 * (self.domain['x'][1] - self.domain['x'][0])
        
        content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
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
    class       dictionary;
    object      snappyHexMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{{geometry_str}
}}

castellatedMeshControls
{{
    maxLocalCells 100000;
    maxGlobalCells 2000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 3;
    
    features
    ({features_str}
    );
    
    refinementSurfaces
    {{{refinement_str}
    }}
    
    resolveFeatureAngle 30;
    
    refinementRegions
    {{
    }}
    
    locationInMesh ({location_x} {y_mid} {z_mid});
    
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 100;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes true;
    layers
    {{
    }}
    expansionRatio 1.0;
    finalLayerThickness 0.3;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}}

meshQualityControls
{{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality -1e30;
    minArea -1;
    minTwist 0.01;
    minDeterminant 0.001;
    minFaceWeight 0.05;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
    relaxed
    {{
        maxNonOrtho 75;
    }}
}}

mergeTolerance 1e-6;

// ************************************************************************* //
'''
        
        with open(self.output_dir / "system" / "snappyHexMeshDict", 'w') as f:
            f.write(content)
    
    def _generate_control_dict(self):
        """Generate controlDict."""
        content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
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
    class       dictionary;
    object      controlDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     {self.solver_type};

startFrom       startTime;

startTime       0;

stopAt          endTime;

endTime         1;

deltaT          1;

writeControl    timeStep;

writeInterval   1;

purgeWrite      0;

writeFormat     ascii;

writePrecision  8;

writeCompression off;

timeFormat      general;

timePrecision   6;

runTimeModifiable true;

functions
{{
    fieldAverage1
    {{
        type            fieldAverage;
        libs            (fieldFunctionObjects);
        writeControl    writeTime;
        fields
        (
            U
            {{
                mean        on;
                prime2Mean  off;
                base        time;
            }}
            p
            {{
                mean        on;
                prime2Mean  off;
                base        time;
            }}
        );
    }}
}}

// ************************************************************************* //
'''
        
        with open(self.output_dir / "system" / "controlDict", 'w') as f:
            f.write(content)
    
    def _generate_fv_schemes(self):
        """Generate fvSchemes."""
        content = '''/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linear;
    div(div(phi,U)) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

// ************************************************************************* //
'''
        
        with open(self.output_dir / "system" / "fvSchemes", 'w') as f:
            f.write(content)
    
    def _generate_fv_solution(self):
        """Generate fvSolution."""
        content = '''/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{
    Phi
    {
        solver          GAMG;
        smoother        DIC;
        tolerance       1e-06;
        relTol          0.01;
    }

    p
    {
        $Phi;
    }
    
    pFinal
    {
        $p;
        relTol          0;
    }
    
    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-06;
        relTol          0.1;
    }
    
    UFinal
    {
        $U;
        relTol          0;
    }
}

potentialFlow
{
    nNonOrthogonalCorrectors 3;
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;
    
    residualControl
    {
        p               1e-4;
        U               1e-4;
    }
}

relaxationFactors
{
    equations
    {
        U               0.9;
        p               0.3;
    }
}

// ************************************************************************* //
'''
        
        with open(self.output_dir / "system" / "fvSolution", 'w') as f:
            f.write(content)
    
    def _generate_boundary_conditions(self):
        """Generate initial/boundary condition files in 0/ directory."""
        # Get velocity components
        vx, vy, vz = self.freestream
        v_mag = self.v_inf
        
        # Build wall patch names from components
        wall_patches = []
        for comp in self.case.scene.components:
            wall_patches.append(comp.name)
        
        wall_bc_U = "\n".join([
            f'''    {name}
    {{
        type            fixedValue;
        value           uniform (0 0 0);
    }}''' for name in wall_patches
        ])
        
        wall_bc_p = "\n".join([
            f'''    {name}
    {{
        type            zeroGradient;
    }}''' for name in wall_patches
        ])
        
        # U (velocity) file
        U_content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
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

internalField   uniform ({vx} {vy} {vz});

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform ({vx} {vy} {vz});
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

{wall_bc_U}
}}

// ************************************************************************* //
'''
        
        # p (pressure) file
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
        type            zeroGradient;
    }}

    bottom
    {{
        type            zeroGradient;
    }}

    front
    {{
        type            symmetry;
    }}

    back
    {{
        type            symmetry;
    }}

{wall_bc_p}
}}

// ************************************************************************* //
'''
        
        with open(self.output_dir / "0" / "U", 'w') as f:
            f.write(U_content)
        
        with open(self.output_dir / "0" / "p", 'w') as f:
            f.write(p_content)
    
    def _generate_transport_properties(self):
        """Generate transportProperties file."""
        # Get viscosity from case if available
        nu = 1e-6  # Default kinematic viscosity (water-like)
        if hasattr(self.case, 'viscosity') and self.case.viscosity is not None:
            # Convert dynamic to kinematic: nu = mu / rho
            nu = self.case.viscosity / self.case.density
        
        content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
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
    class       dictionary;
    object      transportProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

transportModel  Newtonian;

nu              nu [ 0 2 -1 0 0 0 0 ] {nu};

// ************************************************************************* //
'''
        
        with open(self.output_dir / "constant" / "transportProperties", 'w') as f:
            f.write(content)
    
    def _generate_decompose_par_dict(self):
        """Generate decomposeParDict for parallel runs."""
        content = '''/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

numberOfSubdomains 4;

method          scotch;

// ************************************************************************* //
'''
        
        with open(self.output_dir / "system" / "decomposeParDict", 'w') as f:
            f.write(content)
    
    def _generate_sample_dict(self):
        """Generate sampleDict for extracting results on a structured grid."""
        # Use the panel method visualization grid
        x_min, x_max = self.case.x_range
        y_min, y_max = self.case.y_range
        nx, ny = self.case.resolution
        z_mid = self.mesh_settings.z_thickness / 2
        
        content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
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
    class       dictionary;
    object      sampleDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

type sets;

libs (sampling);

interpolationScheme cellPoint;

setFormat raw;

sets
(
);

surfaces
(
    midPlane
    {{
        type            plane;
        planeType       pointAndNormal;
        pointAndNormalDict
        {{
            point       (0 0 {z_mid});
            normal      (0 0 1);
        }}
        interpolate     true;
        triangulate     false;
    }}
    
    // Structured grid matching panel method visualization
    vizGrid
    {{
        type            patch;
        patches         (frontAndBack);
        interpolate     true;
        triangulate     false;
    }}
);

fields
(
    U
    p
);

// ************************************************************************* //
'''
        
        with open(self.output_dir / "system" / "sampleDict", 'w') as f:
            f.write(content)
    
    def _generate_run_scripts(self):
        """Generate Allrun and Allclean scripts."""
        # Component names for mesh
        comp_names = [comp.name for comp in self.case.scene.components]
        
        allrun_content = f'''#!/bin/bash
cd "${{0%/*}}" || exit 1  # Run from this directory

# Source OpenFOAM (adjust path if needed)
# source /opt/openfoam/etc/bashrc  # Uncomment if needed

echo "Running blockMesh..."
blockMesh > log.blockMesh 2>&1

echo "Running surfaceFeatureExtract..."
surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1

echo "Running snappyHexMesh..."
snappyHexMesh -overwrite > log.snappyHexMesh 2>&1

echo "Checking mesh..."
checkMesh > log.checkMesh 2>&1

echo "Running {self.solver_type}..."
{self.solver_type} > log.{self.solver_type} 2>&1

echo "Sampling results..."
postProcess -func sample > log.sample 2>&1

echo "Done!"
'''
        
        allclean_content = '''#!/bin/bash
cd "${0%/*}" || exit 1

# Clean time directories
rm -rf 0.[0-9]* [1-9]* 

# Clean mesh
rm -rf constant/polyMesh

# Clean logs
rm -f log.*

# Clean postProcessing
rm -rf postProcessing

echo "Case cleaned"
'''
        
        # Surface feature extract dict
        sfe_dict_content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
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
    class       dictionary;
    object      surfaceFeatureExtractDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

'''
        for comp_name in comp_names:
            sfe_dict_content += f'''{comp_name}.stl
{{
    extractionMethod    extractFromSurface;
    includedAngle       150;
    subsetFeatures
    {{
        nonManifoldEdges    yes;
        openEdges           yes;
    }}
}}

'''
        
        sfe_dict_content += '// ************************************************************************* //\n'
        
        with open(self.output_dir / "Allrun", 'w') as f:
            f.write(allrun_content)
        
        with open(self.output_dir / "Allclean", 'w') as f:
            f.write(allclean_content)
        
        with open(self.output_dir / "system" / "surfaceFeatureExtractDict", 'w') as f:
            f.write(sfe_dict_content)
        
        # Make scripts executable
        (self.output_dir / "Allrun").chmod(0o755)
        (self.output_dir / "Allclean").chmod(0o755)
