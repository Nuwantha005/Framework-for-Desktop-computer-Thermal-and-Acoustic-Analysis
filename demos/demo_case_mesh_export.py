#!/usr/bin/env python3
"""Export a case mesh to VTK for ParaView."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to import path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.io.case_loader import CaseLoader
from core.geometry.io.vtk_export import export_mesh_vtk


def _require_pyvista() -> None:
    try:
        import pyvista  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PyVista is required for VTK export. Install with `pip install pyvista`."
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Export case mesh to VTK")
    parser.add_argument("case_dir", type=Path, help="Path to case directory")
    parser.add_argument("--mesh-level", type=int, default=0, help="Mesh level index")
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output filename (default: mesh_level_<index>.vtp)",
    )
    args = parser.parse_args()

    _require_pyvista()

    case_dir = args.case_dir.resolve()
    case_yaml = case_dir / "case.yaml"
    if not case_yaml.exists():
        print(f"Error: case.yaml not found at {case_yaml}")
        return 1

    scene, config = CaseLoader.load(case_yaml, mesh_level_index=args.mesh_level)
    mesh = scene.assemble()
    if mesh.dimension != 3:
        print(f"Error: expected 3D mesh, got dimension={mesh.dimension}")
        return 1

    out_dir = case_dir / "out" / "panel_solver"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"mesh_level_{args.mesh_level}.vtp"
    out_path = out_dir / output_name

    export_mesh_vtk(mesh, out_path)

    print(f"Exported mesh for case '{config.name}':")
    print(f"  - {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
