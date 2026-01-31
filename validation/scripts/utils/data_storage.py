"""
Data storage utilities for validation pipeline.

Raw data is stored in /out/raw/ folder as separate files for flexibility:
- Allows custom external visualizations
- Easier to work with individual datasets
- No single large YAML file
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import yaml
import json


def _convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for YAML serialization.
    
    Args:
        obj: Object to convert (dict, list, numpy type, or other)
    
    Returns:
        Object with all numpy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def save_monitoring_point_data(
    output_dir: Path,
    point_data: Dict[str, Dict[str, Dict[str, float]]],
    level_names: List[str],
    study_name: str = "of_convergence"
) -> None:
    """
    Save monitoring point data to CSV files.
    
    Creates:
        raw/monitoring_points_<quantity>.csv
        
    Args:
        output_dir: Base output directory (e.g., cases/case_name/out/)
        point_data: Dict[level_name][point_name][quantity] = value
        level_names: Ordered list of level names
        study_name: Subfolder name (of_convergence, panel_convergence)
    """
    raw_dir = output_dir / study_name / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Organize by quantity
    quantities = set()
    for level_data in point_data.values():
        for point_values in level_data.values():
            quantities.update(point_values.keys())
    
    for quantity in quantities:
        # Build DataFrame: rows = levels, columns = points
        rows = []
        for level in level_names:
            if level not in point_data:
                continue
            row = {'level': level}
            for point_name, values in point_data[level].items():
                row[point_name] = values.get(quantity, np.nan)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(raw_dir / f"monitoring_points_{quantity}.csv", index=False)
    
    # Save metadata
    metadata = {
        'level_names': level_names,
        'quantities': list(quantities),
        'point_names': list(next(iter(point_data.values())).keys()) if point_data else [],
        'study_name': study_name
    }
    with open(raw_dir / "metadata.yaml", 'w') as f:
        yaml.dump(_convert_numpy_types(metadata), f, default_flow_style=False)


def save_convergence_metrics(
    output_dir: Path,
    metrics: Dict[str, Dict[str, Any]],
    study_name: str = "of_convergence"
) -> None:
    """
    Save convergence metrics (GCI, order, etc.) to YAML.
    
    Args:
        output_dir: Base output directory
        metrics: Dict[point_name][quantity] = metrics_dict
        study_name: Subfolder name
    """
    raw_dir = output_dir / study_name / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    with open(raw_dir / "convergence_metrics.yaml", 'w') as f:
        yaml.dump(_convert_numpy_types(metrics), f, default_flow_style=False)


def save_field_data(
    output_dir: Path,
    field_data: Dict[str, np.ndarray],
    level_name: str,
    study_name: str = "of_convergence"
) -> None:
    """
    Save field data (XX, YY, velocity, pressure) to NPZ format.
    
    Args:
        output_dir: Base output directory
        field_data: Dict with XX, YY, Vx, Vy, velocity_magnitude, pressure, etc.
        level_name: Name of refinement level
        study_name: Subfolder name
    """
    raw_dir = output_dir / study_name / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        raw_dir / f"field_data_{level_name}.npz",
        **field_data
    )


def save_surface_data(
    output_dir: Path,
    surface_data: Dict[str, Any],
    solver_name: str,
    of_level: Optional[str] = None
) -> None:
    """
    Save surface comparison data (positions, Vt, Cp per component).
    
    Args:
        output_dir: Base output directory
        surface_data: Dict with component_name -> {s, Vt, Cp, ...}
        solver_name: 'panel' or 'openfoam'
        of_level: OpenFOAM level name if applicable
    """
    if of_level:
        raw_dir = output_dir / "surface_comparison" / f"of_{of_level}" / "raw"
    else:
        raw_dir = output_dir / "surface_comparison" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each component separately
    for comp_name, comp_data in surface_data.items():
        df = pd.DataFrame(comp_data)
        filename = f"{solver_name}_{comp_name}.csv"
        df.to_csv(raw_dir / filename, index=False)


def save_panel_convergence_data(
    output_dir: Path,
    panel_data: Dict[int, Dict[str, Dict[str, float]]],
    reference_data: Dict[str, Dict[str, float]]
) -> None:
    """
    Save panel convergence data (values at different panel counts).
    
    Args:
        output_dir: Base output directory
        panel_data: Dict[panel_count][point_name][quantity] = value
        reference_data: Dict[point_name][quantity] = reference_value
    """
    raw_dir = output_dir / "panel_convergence" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Panel results by quantity
    quantities = set()
    for level_data in panel_data.values():
        for point_values in level_data.values():
            quantities.update(point_values.keys())
    
    for quantity in quantities:
        rows = []
        for panel_count in sorted(panel_data.keys()):
            row = {'panel_count': panel_count}
            for point_name, values in panel_data[panel_count].items():
                row[point_name] = values.get(quantity, np.nan)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(raw_dir / f"panel_values_{quantity}.csv", index=False)
    
    # Reference values
    ref_rows = []
    for point_name, values in reference_data.items():
        row = {'point': point_name}
        row.update(values)
        ref_rows.append(row)
    
    df_ref = pd.DataFrame(ref_rows)
    df_ref.to_csv(raw_dir / "reference_values.csv", index=False)


def save_error_metrics(
    output_dir: Path,
    error_data: Dict[str, Any],
    study_name: str = "surface_comparison",
    of_level: Optional[str] = None
) -> None:
    """
    Save error metrics (L2, Linf, RMS, etc.) to YAML.
    
    Args:
        output_dir: Base output directory
        error_data: Dict with error metrics by component/quantity
        study_name: Subfolder name
        of_level: OpenFOAM level if applicable
    """
    if of_level and study_name == "surface_comparison":
        raw_dir = output_dir / study_name / f"of_{of_level}" / "raw"
    else:
        raw_dir = output_dir / study_name / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    with open(raw_dir / "error_metrics.yaml", 'w') as f:
        yaml.dump(_convert_numpy_types(error_data), f, default_flow_style=False)


def load_monitoring_point_data(
    output_dir: Path,
    study_name: str = "of_convergence"
) -> Dict[str, pd.DataFrame]:
    """
    Load monitoring point data from CSV files.
    
    Returns:
        Dict[quantity] = DataFrame with columns [level, point1, point2, ...]
    """
    raw_dir = output_dir / study_name / "raw"
    
    data = {}
    for csv_file in raw_dir.glob("monitoring_points_*.csv"):
        quantity = csv_file.stem.replace("monitoring_points_", "")
        data[quantity] = pd.read_csv(csv_file)
    
    return data


def load_metadata(
    output_dir: Path,
    study_name: str
) -> Dict[str, Any]:
    """Load metadata from study."""
    raw_dir = output_dir / study_name / "raw"
    metadata_path = raw_dir / "metadata.yaml"
    
    if not metadata_path.exists():
        return {}
    
    with open(metadata_path, 'r') as f:
        return yaml.safe_load(f)


def load_field_data(
    output_dir: Path,
    level_name: str,
    study_name: str = "of_convergence"
) -> Dict[str, np.ndarray]:
    """Load field data from NPZ file."""
    raw_dir = output_dir / study_name / "raw"
    npz_file = raw_dir / f"field_data_{level_name}.npz"
    
    if not npz_file.exists():
        raise FileNotFoundError(f"Field data not found: {npz_file}")
    
    data = np.load(npz_file)
    return {key: data[key] for key in data.files}


def load_convergence_metrics(
    output_dir: Path,
    study_name: str
) -> Dict[str, Any]:
    """Load convergence metrics."""
    raw_dir = output_dir / study_name / "raw"
    metrics_path = raw_dir / "convergence_metrics.yaml"
    
    if not metrics_path.exists():
        return {}
    
    with open(metrics_path, 'r') as f:
        return yaml.safe_load(f)


def load_surface_data(
    output_dir: Path,
    solver_name: str,
    of_level: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load surface data.
    
    Returns:
        Dict[component_name] = DataFrame with s, Vt, Cp, etc.
    """
    if of_level:
        raw_dir = output_dir / "surface_comparison" / f"of_{of_level}" / "raw"
    else:
        raw_dir = output_dir / "surface_comparison" / "raw"
    
    data = {}
    for csv_file in raw_dir.glob(f"{solver_name}_*.csv"):
        comp_name = csv_file.stem.replace(f"{solver_name}_", "")
        data[comp_name] = pd.read_csv(csv_file)
    
    return data


def load_error_metrics(
    output_dir: Path,
    study_name: str = "surface_comparison",
    of_level: Optional[str] = None
) -> Dict[str, Any]:
    """Load error metrics."""
    if of_level and study_name == "surface_comparison":
        raw_dir = output_dir / study_name / f"of_{of_level}" / "raw"
    else:
        raw_dir = output_dir / study_name / "raw"
    
    metrics_path = raw_dir / "error_metrics.yaml"
    
    if not metrics_path.exists():
        return {}
    
    with open(metrics_path, 'r') as f:
        return yaml.safe_load(f)
