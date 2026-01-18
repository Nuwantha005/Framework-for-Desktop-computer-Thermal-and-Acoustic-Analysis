"""
Matplotlib-based visualization for 2D meshes.
"""

from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
import matplotlib.colors as mcolors

from core.geometry import Mesh, Component, Scene


class MeshPlotter:
    """
    2D mesh visualization using matplotlib.
    """
    
    def __init__(self, figsize: Tuple[float, float] = (10, 8)):
        """
        Initialize plotter.
        
        Args:
            figsize: Figure size (width, height) in inches
        """
        self.figsize = figsize
        self.fig = None
        self.ax = None
    
    def _setup_figure(self, title: str = ""):
        """Create figure and axis."""
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        if title:
            self.ax.set_title(title, fontsize=14, fontweight='bold')
    
    def plot_mesh(self, 
                  mesh: Mesh,
                  show_nodes: bool = True,
                  show_panels: bool = True,
                  show_normals: bool = False,
                  show_centers: bool = False,
                  component_colors: bool = True,
                  title: str = "Mesh") -> plt.Figure:
        """
        Plot a mesh.
        
        Args:
            mesh: Mesh to plot
            show_nodes: Draw node markers
            show_panels: Draw panel edges
            show_normals: Draw normal vectors
            show_centers: Draw panel centers
            component_colors: Color panels by component ID
            title: Plot title
        
        Returns:
            Matplotlib figure
        """
        if not mesh.is_2d:
            raise ValueError("MeshPlotter only supports 2D meshes")
        
        self._setup_figure(title)
        
        # Get unique components
        component_ids = np.unique(mesh.component_ids)
        
        # Color map for components
        if component_colors and len(component_ids) > 1:
            colors = plt.cm.tab10(np.linspace(0, 1, len(component_ids)))
            color_map = {cid: colors[i] for i, cid in enumerate(component_ids)}
        else:
            color_map = {cid: 'black' for cid in component_ids}
        
        # Plot panels
        if show_panels:
            for i in range(mesh.num_panels):
                n1_idx, n2_idx = mesh.panels[i]
                p1 = mesh.nodes[n1_idx, :2]
                p2 = mesh.nodes[n2_idx, :2]
                
                comp_id = mesh.component_ids[i]
                color = color_map[comp_id]
                
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                           color=color, linewidth=2, marker='')
        
        # Plot nodes
        if show_nodes:
            self.ax.plot(mesh.nodes[:, 0], mesh.nodes[:, 1], 
                        'ko', markersize=4, zorder=10, label='Nodes')
        
        # Plot panel centers
        if show_centers and mesh.centers is not None:
            self.ax.plot(mesh.centers[:, 0], mesh.centers[:, 1],
                        'rx', markersize=6, zorder=11, label='Panel Centers')
        
        # Plot normals
        if show_normals and mesh.normals is not None and mesh.centers is not None:
            scale = 0.2  # Normal vector length
            for i in range(mesh.num_panels):
                center = mesh.centers[i, :2]
                normal = mesh.normals[i, :2]
                
                # Draw arrow
                self.ax.arrow(center[0], center[1],
                            scale * normal[0], scale * normal[1],
                            head_width=0.05, head_length=0.05,
                            fc='red', ec='red', alpha=0.7, zorder=12)
        
        # Add legend if multiple components
        if component_colors and len(component_ids) > 1:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color=color_map[cid], linewidth=2, 
                      label=f'Component {cid}')
                for cid in component_ids
            ]
            if show_nodes:
                legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor='k', markersize=6,
                                             label='Nodes'))
            if show_normals:
                legend_elements.append(Line2D([0], [0], color='red',
                                             marker='>', markersize=8,
                                             label='Normals'))
            self.ax.legend(handles=legend_elements, loc='best')
        
        # Auto-scale with padding
        self._auto_scale_axis(mesh.nodes[:, :2], padding=0.5)
        
        return self.fig
    
    def plot_component(self,
                      component: Component,
                      show_local: bool = True,
                      show_global: bool = True,
                      show_normals: bool = True,
                      title: Optional[str] = None) -> plt.Figure:
        """
        Plot a component showing both local and global coordinates.
        
        Args:
            component: Component to plot
            show_local: Show local coordinate mesh
            show_global: Show transformed (global) mesh
            show_normals: Show normal vectors
            title: Plot title (default: component name)
        
        Returns:
            Matplotlib figure
        """
        if title is None:
            title = f"Component: {component.name}"
        
        self._setup_figure(title)
        
        # Plot local mesh
        if show_local:
            local_mesh = component.local_mesh
            for i in range(local_mesh.num_panels):
                n1_idx, n2_idx = local_mesh.panels[i]
                p1 = local_mesh.nodes[n1_idx, :2]
                p2 = local_mesh.nodes[n2_idx, :2]
                
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                           'b--', linewidth=1.5, alpha=0.5, label='Local' if i == 0 else '')
            
            # Local origin marker
            self.ax.plot(0, 0, 'bx', markersize=10, markeredgewidth=2,
                       label='Local Origin')
        
        # Plot global (transformed) mesh
        if show_global:
            global_mesh = component.get_global_mesh(component_id=0)
            for i in range(global_mesh.num_panels):
                n1_idx, n2_idx = global_mesh.panels[i]
                p1 = global_mesh.nodes[n1_idx, :2]
                p2 = global_mesh.nodes[n2_idx, :2]
                
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                           'r-', linewidth=2, label='Global' if i == 0 else '')
            
            # Plot normals on global mesh
            if show_normals and global_mesh.normals is not None:
                scale = 0.2
                for i in range(global_mesh.num_panels):
                    center = global_mesh.centers[i, :2]
                    normal = global_mesh.normals[i, :2]
                    
                    self.ax.arrow(center[0], center[1],
                                scale * normal[0], scale * normal[1],
                                head_width=0.05, head_length=0.05,
                                fc='darkred', ec='darkred', alpha=0.7)
            
            # Global origin marker (transform translation)
            global_origin = component.transform.translation[:2]
            self.ax.plot(global_origin[0], global_origin[1], 
                       'r+', markersize=12, markeredgewidth=2,
                       label='Global Origin')
        
        self.ax.legend(loc='best')
        
        # Auto-scale
        all_nodes = []
        if show_local:
            all_nodes.append(component.local_mesh.nodes[:, :2])
        if show_global:
            global_mesh = component.get_global_mesh(component_id=0)
            all_nodes.append(global_mesh.nodes[:, :2])
        
        if all_nodes:
            all_nodes = np.vstack(all_nodes)
            self._auto_scale_axis(all_nodes, padding=0.5)
        
        return self.fig
    
    def plot_scene(self,
                   scene: Scene,
                   show_normals: bool = False,
                   show_centers: bool = False,
                   show_freestream: bool = True,
                   title: Optional[str] = None) -> plt.Figure:
        """
        Plot entire scene with all components.
        
        Args:
            scene: Scene to plot
            show_normals: Show panel normals
            show_centers: Show panel centers
            show_freestream: Show freestream velocity arrow
            title: Plot title (default: scene name)
        
        Returns:
            Matplotlib figure
        """
        if title is None:
            title = f"Scene: {scene.name}"
        
        # Assemble global mesh
        global_mesh = scene.assemble()
        
        # Plot mesh with component colors
        self.plot_mesh(
            global_mesh,
            show_nodes=True,
            show_panels=True,
            show_normals=show_normals,
            show_centers=show_centers,
            component_colors=True,
            title=title
        )
        
        # Add freestream arrow
        if show_freestream:
            # Place arrow in upper-left corner
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            arrow_x = xlim[0] + 0.1 * (xlim[1] - xlim[0])
            arrow_y = ylim[1] - 0.1 * (ylim[1] - ylim[0])
            
            # Normalize and scale freestream
            fs_norm = scene.freestream[:2] / (np.linalg.norm(scene.freestream[:2]) + 1e-10)
            arrow_length = 0.15 * (xlim[1] - xlim[0])
            
            self.ax.arrow(arrow_x, arrow_y,
                        arrow_length * fs_norm[0],
                        arrow_length * fs_norm[1],
                        head_width=0.15, head_length=0.15,
                        fc='blue', ec='blue', alpha=0.8, linewidth=2,
                        length_includes_head=True, zorder=15)
            
            # Label
            self.ax.text(arrow_x, arrow_y + 0.05 * (ylim[1] - ylim[0]),
                       f'V∞ = ({scene.freestream[0]:.2f}, {scene.freestream[1]:.2f})',
                       fontsize=10, color='blue', fontweight='bold',
                       ha='left', va='bottom')
        
        # Add info text
        info_text = (
            f"Components: {scene.num_components}\n"
            f"Total Panels: {global_mesh.num_panels}\n"
            f"Total Nodes: {global_mesh.num_nodes}"
        )
        
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.text(xlim[1] - 0.02 * (xlim[1] - xlim[0]),
                   ylim[0] + 0.02 * (ylim[1] - ylim[0]),
                   info_text,
                   fontsize=9, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return self.fig
    
    def _auto_scale_axis(self, points: np.ndarray, padding: float = 0.5):
        """Auto-scale axis with padding."""
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        if x_range < 1e-10:
            x_range = 1.0
        if y_range < 1e-10:
            y_range = 1.0
        
        self.ax.set_xlim(x_min - padding, x_max + padding)
        self.ax.set_ylim(y_min - padding, y_max + padding)
    
    def save(self, filepath: str, dpi: int = 300):
        """
        Save figure to file.
        
        Args:
            filepath: Output file path
            dpi: Resolution (dots per inch)
        """
        if self.fig is None:
            raise ValueError("No figure to save. Call a plot method first.")
        
        self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved figure to: {filepath}")
    
    def show(self):
        """Display the figure."""
        if self.fig is None:
            raise ValueError("No figure to show. Call a plot method first.")
        
        plt.show()
    
    def close(self):
        """Close the figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


def quick_plot_mesh(mesh: Mesh, show_normals: bool = True):
    """Quick plot of a mesh (convenience function)."""
    plotter = MeshPlotter()
    plotter.plot_mesh(mesh, show_normals=show_normals)
    plotter.show()


def quick_plot_component(component: Component, show_normals: bool = True):
    """Quick plot of a component (convenience function)."""
    plotter = MeshPlotter()
    plotter.plot_component(component, show_normals=show_normals)
    plotter.show()


def quick_plot_scene(scene: Scene, show_normals: bool = False):
    """Quick plot of a scene (convenience function)."""
    plotter = MeshPlotter()
    plotter.plot_scene(scene, show_normals=show_normals)
    plotter.show()
