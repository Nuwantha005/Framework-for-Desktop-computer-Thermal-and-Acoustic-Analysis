"""
Configuration file loaders and validators for validation scripts.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml


def load_of_config(config_path: Path) -> Dict[str, Any]:
    """
    Load OpenFOAM convergence configuration from of_case/config.yaml.
    
    Expected structure:
        refinement_levels:
          - name: "coarse"
            blockMesh_cells: [100, 80, 1]
            components:
              component_name:
                surface_level: 2
                feature_level: 2
        
        monitoring_points:
          - name: "point1"
            coordinates: [x, y]
        
        convergence:
          gci_threshold: 0.05
          refinement_ratio: 1.5
    
    Args:
        config_path: Path to config.yaml file
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required = ['refinement_levels', 'monitoring_points']
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in {config_path}")
    
    # Validate refinement levels
    if not config['refinement_levels']:
        raise ValueError("At least one refinement level must be defined")
    
    for level in config['refinement_levels']:
        if 'name' not in level:
            raise ValueError("Each refinement level must have a 'name'")
        if 'blockMesh_cells' not in level:
            raise ValueError(f"Refinement level '{level['name']}' missing 'blockMesh_cells'")
        if len(level['blockMesh_cells']) != 3:
            raise ValueError(f"blockMesh_cells must have 3 values [nx, ny, nz]")
    
    # Validate monitoring points
    if not config['monitoring_points']:
        raise ValueError("At least one monitoring point must be defined")
    
    for point in config['monitoring_points']:
        if 'name' not in point or 'coordinates' not in point:
            raise ValueError("Each monitoring point must have 'name' and 'coordinates'")
        if len(point['coordinates']) != 2:
            raise ValueError(f"Point '{point['name']}' coordinates must be [x, y]")
    
    return config


def load_viz_config(config_path: Path) -> Dict[str, Any]:
    """
    Load visualization configuration from out/viz_config.yaml.
    
    Returns default configuration if file doesn't exist.
    
    Args:
        config_path: Path to viz_config.yaml file
    
    Returns:
        Visualization configuration dictionary with defaults filled in
    """
    # Default configuration
    default_config = {
        'figure': {
            'dpi': 300,
            'format': ['png'],
            'style': 'paper',
        },
        'of_convergence': {
            'field_overview': {
                'enabled': True,
                'field': 'velocity_magnitude',
                'show_streamlines': True,
                'show_monitoring_points': True,
                'point_marker_size': 100,
                'colorbar': True,
            },
            'convergence_curves': {
                'enabled': True,
                'quantities': ['velocity', 'pressure'],
                'x_axis': 'level_name',
                'show_change_rate': True,
                'log_scale': False,
            },
            'per_point_plots': {
                'enabled': True,
            },
            'combined_points_plot': {
                'enabled': True,
                'normalize': False,
            },
            'table_output': {
                'enabled': True,
                'format': ['csv'],
            },
        },
        'panel_convergence': {
            'reference_of_level': None,  # Will be set to finest by default
            'convergence_curves': {
                'enabled': True,
                'show_of_reference_line': True,
                'of_line_style': '--',
                'of_line_color': 'red',
                'quantities': ['velocity', 'pressure'],
            },
            'error_vs_panels': {
                'enabled': True,
                'error_metric': 'relative',
            },
        },
        'surface_comparison': {
            'quantities': ['Vt', 'Cp'],
            'show_by_component': False,
            'error_metrics': True,
            'interpolation_method': 'linear',
        },
    }
    
    # Load user config if exists
    if config_path.exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
        
        # Deep merge user config into defaults
        if user_config:
            default_config = _deep_merge(default_config, user_config)
    
    return default_config


def _deep_merge(base: Dict, update: Dict) -> Dict:
    """Recursively merge update dict into base dict."""
    result = base.copy()
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def save_viz_config(config: Dict[str, Any], config_path: Path) -> None:
    """
    Save visualization configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config file
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def create_default_of_config(
    case_name: str,
    num_levels: int = 4,
    component_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create a default OpenFOAM convergence configuration.
    
    Args:
        case_name: Name of the case
        num_levels: Number of refinement levels to generate
        component_names: List of component names (if None, uses generic settings)
    
    Returns:
        Configuration dictionary
    """
    # Generate refinement levels with increasing resolution
    base_cells = [80, 60, 1]
    refinement_levels = []
    level_names = ['coarse', 'medium', 'fine', 'very_fine', 'ultra_fine']
    
    for i in range(num_levels):
        factor = 1.0 + i * 0.3  # Increase by 30% each level
        cells = [int(base_cells[0] * factor), int(base_cells[1] * factor), 1]
        surface_level = 2 + i
        
        level_config = {
            'name': level_names[i] if i < len(level_names) else f'level_{i}',
            'blockMesh_cells': cells,
        }
        
        # Add component-specific settings if provided
        if component_names:
            level_config['components'] = {}
            for comp_name in component_names:
                level_config['components'][comp_name] = {
                    'surface_level': surface_level,
                    'feature_level': surface_level,
                }
        else:
            # Global snappy settings
            level_config['snappy_surface_level'] = surface_level
            level_config['snappy_feature_level'] = surface_level
        
        refinement_levels.append(level_config)
    
    # Default monitoring points (downstream locations)
    monitoring_points = [
        {'name': 'upstream', 'coordinates': [-2.0, 0.0]},
        {'name': 'downstream_near', 'coordinates': [2.0, 0.0]},
        {'name': 'downstream_far', 'coordinates': [4.0, 0.0]},
    ]
    
    config = {
        'refinement_levels': refinement_levels,
        'monitoring_points': monitoring_points,
        'convergence': {
            'gci_threshold': 0.05,
            'refinement_ratio': 1.3,
        },
        'output': {
            'delete_intermediate_cases': False,
            'save_final_case': True,
            'verbose': True,
        },
    }
    
    return config
