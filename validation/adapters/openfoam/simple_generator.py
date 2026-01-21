"""
Simple OpenFOAM case generator using blockMesh only.

For simple cases where snappyHexMesh is not needed (external flow around domain
boundaries rather than internal bodies). Useful for:
- Initial testing
- Cases where the panel method bodies define the domain boundaries
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np


@dataclass
class SimpleBlockMeshSettings:
    """Settings for simple blockMesh case."""
    cells_per_unit: float = 20.0
    z_thickness: float = 0.1
    grading_x: float = 1.0
    grading_y: float = 1.0


class SimpleOpenFOAMGenerator:
    """
    Generate simple OpenFOAM case using blockMesh only (no snappyHexMesh).
    
    This is useful for:
    - Quick testing of the validation pipeline
    - External flow cases where domain bounds are the boundaries
    - Comparison of free-stream behavior
    
    For cases with internal bodies, use OpenFOAMCaseGenerator instead.
    
    Usage:
        generator = SimpleOpenFOAMGenerator(
            x_range=(-2, 8),
            y_range=(-3, 3),
            freestream=(10, 0, 0),
            output_dir="validation_results/simple_test"
        )
        generator.generate()
    """
    
    def __init__(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        freestream: Tuple[float, float, float],
        output_dir: Path | str,
        settings: Optional[SimpleBlockMeshSettings] = None
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.freestream = freestream
        self.output_dir = Path(output_dir)
        self.settings = settings or SimpleBlockMeshSettings()
        
        self.v_inf = np.linalg.norm(freestream)
    
    def generate(self) -> Path:
        """Generate complete OpenFOAM case."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "0").mkdir(exist_ok=True)
        (self.output_dir / "constant").mkdir(exist_ok=True)
        (self.output_dir / "system").mkdir(exist_ok=True)
        
        self._generate_blockmesh_dict()
        self._generate_control_dict()
        self._generate_fv_schemes()
        self._generate_fv_solution()
        self._generate_boundary_conditions()
        self._generate_transport_properties()
        self._generate_run_scripts()
        
        return self.output_dir
    
    def _generate_blockmesh_dict(self):
        """Generate blockMeshDict."""
        x0, x1 = self.x_range
        y0, y1 = self.y_range
        z0, z1 = 0.0, self.settings.z_thickness
        
        nx = int((x1 - x0) * self.settings.cells_per_unit)
        ny = int((y1 - y0) * self.settings.cells_per_unit)
        
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
    ({x0} {y0} {z0})
    ({x1} {y0} {z0})
    ({x1} {y1} {z0})
    ({x0} {y1} {z0})
    ({x0} {y0} {z1})
    ({x1} {y0} {z1})
    ({x1} {y1} {z1})
    ({x0} {y1} {z1})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} 1) 
    simpleGrading ({self.settings.grading_x} {self.settings.grading_y} 1)
);

boundary
(
    inlet
    {{
        type patch;
        faces ((0 4 7 3));
    }}
    outlet
    {{
        type patch;
        faces ((1 2 6 5));
    }}
    top
    {{
        type patch;
        faces ((3 7 6 2));
    }}
    bottom
    {{
        type patch;
        faces ((0 1 5 4));
    }}
    frontAndBack
    {{
        type empty;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);

// ************************************************************************* //
'''
        with open(self.output_dir / "system" / "blockMeshDict", 'w') as f:
            f.write(content)
    
    def _generate_control_dict(self):
        """Generate controlDict."""
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
    object      controlDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     potentialFoam;

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
}

potentialFlow
{
    nNonOrthogonalCorrectors 3;
}

// ************************************************************************* //
'''
        with open(self.output_dir / "system" / "fvSolution", 'w') as f:
            f.write(content)
    
    def _generate_boundary_conditions(self):
        """Generate boundary condition files."""
        vx, vy, vz = self.freestream
        
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
    frontAndBack
    {{
        type            empty;
    }}
}}

// ************************************************************************* //
'''
        
        p_content = '''/*--------------------------------*- C++ -*----------------------------------*\\
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
    class       volScalarField;
    object      p;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    top
    {
        type            zeroGradient;
    }
    bottom
    {
        type            zeroGradient;
    }
    frontAndBack
    {
        type            empty;
    }
}

// ************************************************************************* //
'''
        
        with open(self.output_dir / "0" / "U", 'w') as f:
            f.write(U_content)
        
        with open(self.output_dir / "0" / "p", 'w') as f:
            f.write(p_content)
    
    def _generate_transport_properties(self):
        """Generate transportProperties."""
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
    object      transportProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

transportModel  Newtonian;

nu              nu [ 0 2 -1 0 0 0 0 ] 1e-06;

// ************************************************************************* //
'''
        with open(self.output_dir / "constant" / "transportProperties", 'w') as f:
            f.write(content)
    
    def _generate_run_scripts(self):
        """Generate Allrun and Allclean scripts."""
        allrun = '''#!/bin/bash
cd "${0%/*}" || exit 1

echo "Running blockMesh..."
blockMesh > log.blockMesh 2>&1

echo "Checking mesh..."
checkMesh > log.checkMesh 2>&1

echo "Running potentialFoam..."
potentialFoam > log.potentialFoam 2>&1

echo "Done!"
'''
        
        allclean = '''#!/bin/bash
cd "${0%/*}" || exit 1

rm -rf [1-9]* 0.[0-9]*
rm -rf constant/polyMesh
rm -f log.*
rm -rf postProcessing

echo "Case cleaned"
'''
        
        with open(self.output_dir / "Allrun", 'w') as f:
            f.write(allrun)
        
        with open(self.output_dir / "Allclean", 'w') as f:
            f.write(allclean)
        
        (self.output_dir / "Allrun").chmod(0o755)
        (self.output_dir / "Allclean").chmod(0o755)
