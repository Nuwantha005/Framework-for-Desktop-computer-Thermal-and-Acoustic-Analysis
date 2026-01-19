# Demos

Demo scripts that operate on case files in `../cases/`.

## Available Demos

| Script | Description |
|--------|-------------|
| `demo_mesh_plot.py` | Plot mesh geometry from a case |
| `demo_streamlines.py` | Solve and plot streamlines |
| `demo_contours.py` | Solve and plot velocity contours |
| `demo_combined.py` | Combined 2×2 subplot (mesh, contours, streamlines, Cp) |

## Usage

```bash
# Navigate to demos folder
cd demos

# Plot mesh with normals
python demo_mesh_plot.py ../cases/cylinder_flow --save --normals

# Show streamlines interactively
python demo_streamlines.py ../cases/cylinder_flow --show

# Save combined plot with 6 cores
python demo_combined.py ../cases/single_square --save --cores 6

# Save to timestamped subfolder (avoids overwriting)
python demo_combined.py ../cases/cylinder_flow --save --protect
```

## Arguments

- `case_dir`: Path to case folder (must contain `case.yaml`)
- `--show`: Display plot interactively
- `--save`: Save to `case_dir/out/` (default if neither specified)
- `--protect`: Save to timestamped subfolder
- `--cores N`: Number of CPU cores (default: 6)
- `--normals`: Show panel normals (mesh_plot only)
- `--levels N`: Contour levels (contours only)
