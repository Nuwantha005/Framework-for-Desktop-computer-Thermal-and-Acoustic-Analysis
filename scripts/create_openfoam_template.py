#!/usr/bin/env python3
"""
Create OpenFOAM template case for potentialFoam 2D simulations.

This template will be cloned and modified for each validation case.
"""

from pathlib import Path
from foamlib import FoamCase, FoamFile, FoamFieldFile, Dimensioned, DimensionSet

# Template location (two levels up from scripts/ → project root)
template_dir = Path(__file__).parent.parent / "templates" / "openfoam" / "potentialFoam2D"
template_dir.mkdir(parents=True, exist_ok=True)

# Create directory structure
(template_dir / "0").mkdir(exist_ok=True)
(template_dir / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)
(template_dir / "system").mkdir(exist_ok=True)

print(f"Creating template in: {template_dir}")

# Use foamlib to create a minimal case
case = FoamCase(template_dir)

# ============================================================================
# 0/U - Velocity field
# ============================================================================
print("  Creating 0/U...")
with case[0]["U"] as U:
    U.dimensions = DimensionSet(length=1, time=-1)  # [0 1 -1 0 0 0 0]
    U.internal_field = [10, 0, 0]
    U.boundary_field = {
        "inlet":  {"type": "fixedValue", "value": [10, 0, 0]},
        "outlet": {"type": "zeroGradient"},
        "top":    {"type": "slip"},
        "bottom": {"type": "slip"},
        "front":  {"type": "symmetry"},
        "back":   {"type": "symmetry"},
    }

# ============================================================================
# 0/p - Pressure field
# ============================================================================
print("  Creating 0/p...")
with case[0]["p"] as p:
    p.dimensions = DimensionSet(length=2, time=-2)  # [0 2 -2 0 0 0 0]
    p.internal_field = 0
    p.boundary_field = {
        "inlet":  {"type": "zeroGradient"},
        "outlet": {"type": "fixedValue", "value": 0},
        "top":    {"type": "zeroGradient"},
        "bottom": {"type": "zeroGradient"},
        "front":  {"type": "symmetry"},
        "back":   {"type": "symmetry"},
    }

# ============================================================================
# constant/transportProperties
# ============================================================================
print("  Creating constant/transportProperties...")
with case.transport_properties as f:
    f["transportModel"] = "Newtonian"
    f["nu"] = Dimensioned(1.5e-5, DimensionSet(length=2, time=-1), "nu")

# ============================================================================
# system/controlDict
# ============================================================================
print("  Creating system/controlDict...")
with case.control_dict as f:
    f["application"] = "potentialFoam"
    f["startFrom"] = "startTime"
    f["startTime"] = 0
    f["stopAt"] = "endTime"
    f["endTime"] = 1
    f["deltaT"] = 1
    f["writeControl"] = "timeStep"
    f["writeInterval"] = 1
    f["purgeWrite"] = 0
    f["writeFormat"] = "ascii"
    f["writePrecision"] = 6
    f["writeCompression"] = False
    f["timeFormat"] = "general"
    f["timePrecision"] = 6
    f["runTimeModifiable"] = True

# ============================================================================
# system/fvSchemes
# ============================================================================
print("  Creating system/fvSchemes...")
with case.fv_schemes as f:
    f["ddtSchemes"] = {"default": "steadyState"}
    f["gradSchemes"] = {"default": ("Gauss", "linear")}
    f["divSchemes"] = {
        "default": "none",
        "div(phi,U)": ("bounded", "Gauss", "linear"),
        "div(div(phi,U))": ("Gauss", "linear"),
    }
    f["laplacianSchemes"] = {"default": ("Gauss", "linear", "corrected")}
    f["interpolationSchemes"] = {"default": "linear"}
    f["snGradSchemes"] = {"default": "corrected"}

# ============================================================================
# system/fvSolution
# ============================================================================
print("  Creating system/fvSolution...")
# Write manually due to foamlib limitations with variable references ($p, $Phi)
fv_solution_content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}

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

    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-06;
        relTol          0.1;
    }
}

potentialFlow
{
    nNonOrthogonalCorrectors 3;
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent          yes;
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
"""
(template_dir / "system" / "fvSolution").write_text(fv_solution_content)

# ============================================================================
# system/blockMeshDict
# ============================================================================
print("  Creating system/blockMeshDict...")
with case.block_mesh_dict as f:
    f["scale"] = 1
    
    # Placeholder vertices (will be updated by generator)
    f["vertices"] = [
        [-5, -3, 0],  # 0
        [ 5, -3, 0],  # 1
        [ 5,  3, 0],  # 2
        [-5,  3, 0],  # 3
        [-5, -3, 0.1],  # 4
        [ 5, -3, 0.1],  # 5
        [ 5,  3, 0.1],  # 6
        [-5,  3, 0.1],  # 7
    ]
    
    f["blocks"] = [
        "hex",
        [0, 1, 2, 3, 4, 5, 6, 7],
        [100, 60, 1],  # Placeholder cell counts
        "simpleGrading",
        [1, 1, 1],
    ]
    
    f["edges"] = []
    
    f["boundary"] = [
        ("inlet",  {"type": "patch", "faces": [[0, 4, 7, 3]]}),
        ("outlet", {"type": "patch", "faces": [[1, 2, 6, 5]]}),
        ("top",    {"type": "patch", "faces": [[3, 7, 6, 2]]}),
        ("bottom", {"type": "patch", "faces": [[0, 1, 5, 4]]}),
        ("front",  {"type": "symmetry", "faces": [[0, 3, 2, 1]]}),
        ("back",   {"type": "symmetry", "faces": [[4, 5, 6, 7]]}),
    ]
    
    f["mergePatchPairs"] = []

# ============================================================================
# system/snappyHexMeshDict
# ============================================================================
print("  Creating system/snappyHexMeshDict...")
snappy = case.file("system/snappyHexMeshDict")
with snappy as f:
    f["castellatedMesh"] = True
    f["snap"] = True
    f["addLayers"] = False
    
    # Placeholder geometry (will be populated by generator)
    f["geometry"] = {}
    
    f["castellatedMeshControls"] = {
        "maxLocalCells": 100000,
        "maxGlobalCells": 2000000,
        "minRefinementCells": 0,
        "maxLoadUnbalance": 0.10,
        "nCellsBetweenLevels": 3,
        "features": [],  # Will be populated
        "refinementSurfaces": {},  # Will be populated
        "resolveFeatureAngle": 30,
        "refinementRegions": {},
        "locationInMesh": [0.001, 0.001, 0.05],  # Placeholder
        "allowFreeStandingZoneFaces": True,
    }
    
    f["snapControls"] = {
        "nSmoothPatch": 3,
        "tolerance": 2.0,
        "nSolveIter": 100,
        "nRelaxIter": 5,
    }
    
    f["addLayersControls"] = {
        "relativeSizes": True,
        "layers": {},
        "expansionRatio": 1.2,
        "finalLayerThickness": 0.3,
        "minThickness": 0.1,
        "nGrow": 0,
        "featureAngle": 60,
        "nRelaxIter": 3,
        "nSmoothSurfaceNormals": 1,
        "nSmoothNormals": 3,
        "nSmoothThickness": 10,
        "maxFaceThicknessRatio": 0.5,
        "maxThicknessToMedialRatio": 0.3,
        "minMedialAxisAngle": 90,
        "nBufferCellsNoExtrude": 0,
        "nLayerIter": 50,
    }
    
    f["meshQualityControls"] = {
        "maxNonOrtho": 65,
        "maxBoundarySkewness": 20,
        "maxInternalSkewness": 4,
        "maxConcave": 80,
        "minVol": 1e-13,
        "minTetQuality": 1e-15,
        "minArea": -1,
        "minTwist": 0.02,
        "minDeterminant": 0.001,
        "minFaceWeight": 0.02,
        "minVolRatio": 0.01,
        "minTriangleTwist": -1,
        "nSmoothScale": 4,
        "errorReduction": 0.75,
    }
    
    f["writeFlags"] = ["scalarLevels", "layerSets", "layerFields"]
    f["mergeTolerance"] = 1e-6

# ============================================================================
# system/surfaceFeatureExtractDict
# ============================================================================
print("  Creating system/surfaceFeatureExtractDict...")
sfe = case.file("system/surfaceFeatureExtractDict")
with sfe as f:
    # Placeholder - will be populated with STL files by generator
    pass

# ============================================================================
# system/decomposeParDict
# ============================================================================
print("  Creating system/decomposeParDict...")
with case.decompose_par_dict as f:
    f["numberOfSubdomains"] = 4
    f["method"] = "scotch"

print(f"\n✓ Template created successfully in: {template_dir}")
print("\nFiles created:")
for file in sorted(template_dir.rglob("*")):
    if file.is_file():
        print(f"  {file.relative_to(template_dir.parent)}")
