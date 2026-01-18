# Demos

Demonstration scripts for various features of the panel method solver.

## Running Demos

From project root:
```bash
python demos/demo_visualization.py
```

Or use the launcher:
```bash
python run_demo.py visualization
```

## Available Demos

- **demo_visualization.py** - Shows mesh, component, and scene visualization capabilities

## Adding New Demos

Place new demo files in this directory with naming convention: `demo_*.py`

All demos should:
1. Add `src/` to Python path
2. Import from `core.*` and other modules
3. Include clear print statements explaining what's happening
4. Close plot windows to continue (for visualization demos)
