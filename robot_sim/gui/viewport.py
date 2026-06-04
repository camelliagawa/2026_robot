"""
3D Viewport for the robot simulation using matplotlib.

Renders:
  - Robot arm as a chain of line segments (joint positions from FK)
  - Knife blade as a thin rectangle at the end-effector
  - Route waypoints as scatter points
  - Route path as a connected line
"""
from __future__ import annotations

from typing import Optional, List, TYPE_CHECKING

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tkinter as tk

if TYPE_CHECKING:
    from ..path.route import Route, Waypoint
    from ..robot.kinematics import Kinematics


# Visual constants
ROBOT_COLOR = "#F5C400"      # FANUC yellow
ROBOT_JOINT_COLOR = "#333333"
KNIFE_BLADE_COLOR = "#C0C0C8"
KNIFE_HANDLE_COLOR = "#3A2010"
ROUTE_PATH_COLOR = "#2288FF"
WAYPOINT_COLOR = "#FF4422"
ACTIVE_WAYPOINT_COLOR = "#00FF88"
GRID_ALPHA = 0.25

# Knife geometry (mm)
KNIFE_HANDLE_LEN = 150.0
KNIFE_BLADE_LEN = 200.0
KNIFE_BLADE_WIDTH = 45.0   # spine to edge
KNIFE_BLADE_THICK = 3.0


class Viewport3D:
    """
    Embedded 3D matplotlib viewport inside a tkinter frame.
    """

    def __init__(self, parent: tk.Widget, kinematics: "Kinematics"):
        """
        Args:
            parent     : Parent tkinter widget.
            kinematics : Kinematics instance for FK computation.
        """
        self.kin = kinematics
        self._route: Optional["Route"] = None
        self._selected_wp_idx: Optional[int] = None
        self._joint_angles = np.zeros(6)

        # Create figure
        self.fig = plt.figure(figsize=(7, 6), facecolor="#1A1A1A")
        self.ax: Axes3D = self.fig.add_subplot(111, projection="3d")
        self._setup_axes()

        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Toolbar
        toolbar_frame = tk.Frame(parent)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

        # Initial draw
        self.update_robot(self._joint_angles)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update_robot(self, joint_angles: np.ndarray):
        """Redraw robot at given joint angles."""
        self._joint_angles = np.asarray(joint_angles)
        self._redraw()

    def set_route(self, route: Optional["Route"]):
        """Set the route to display."""
        self._route = route
        self._redraw()

    def set_selected_waypoint(self, idx: Optional[int]):
        """Highlight a specific waypoint."""
        self._selected_wp_idx = idx
        self._redraw()

    def refresh(self):
        """Force redraw."""
        self._redraw()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _redraw(self):
        """Clear and redraw everything."""
        self.ax.cla()
        self._setup_axes()
        self._draw_workspace_sphere()
        self._draw_robot(self._joint_angles)
        self._draw_route()
        self.canvas.draw_idle()

    def _setup_axes(self):
        """Configure 3D axes appearance."""
        ax = self.ax
        ax.set_facecolor("#1A1A1A")

        lim = 700
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(0, lim * 1.5)

        ax.set_xlabel("X (mm)", color="white", fontsize=8)
        ax.set_ylabel("Y (mm)", color="white", fontsize=8)
        ax.set_zlabel("Z (mm)", color="white", fontsize=8)
        ax.tick_params(colors="gray", labelsize=7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#333333")
        ax.yaxis.pane.set_edgecolor("#333333")
        ax.zaxis.pane.set_edgecolor("#333333")
        ax.grid(True, alpha=GRID_ALPHA, color="#555555")

        # Base platform
        theta = np.linspace(0, 2 * np.pi, 30)
        r = 80
        ax.plot(
            r * np.cos(theta), r * np.sin(theta), np.zeros(30),
            color="#555555", lw=1.5, alpha=0.5
        )

    def _draw_workspace_sphere(self):
        """Draw a translucent sphere indicating approximate workspace."""
        # Just draw a circle at reach radius in XY plane
        reach = 911
        theta = np.linspace(0, 2 * np.pi, 60)
        # Horizontal circle at mid-height
        self.ax.plot(
            reach * np.cos(theta), reach * np.sin(theta),
            np.full(60, 330),
            color="#334455", lw=0.8, alpha=0.3, linestyle="--"
        )

    def _draw_robot(self, q: np.ndarray):
        """Draw robot links and joints."""
        positions = self.kin.get_joint_positions(q)  # (7, 3)

        # Link segments
        xs = positions[:, 0]
        ys = positions[:, 1]
        zs = positions[:, 2]

        # Draw links
        self.ax.plot(xs, ys, zs,
                     color=ROBOT_COLOR, lw=4, solid_capstyle="round",
                     zorder=5)

        # Draw joint markers
        for i, (x, y, z) in enumerate(positions):
            color = ROBOT_JOINT_COLOR if i > 0 else "#888888"
            size = 60 if i == 0 else 40
            self.ax.scatter([x], [y], [z], c=color, s=size, zorder=6, depthshade=False)

        # Draw knife at end-effector
        self._draw_knife(q)

        # Draw EE frame axes
        T_ee = self.kin.forward(q)
        origin = T_ee[:3, 3]
        R = T_ee[:3, :3]
        scale = 60
        for col, color in enumerate(["red", "green", "blue"]):
            axis = origin + scale * R[:, col]
            self.ax.plot(
                [origin[0], axis[0]],
                [origin[1], axis[1]],
                [origin[2], axis[2]],
                color=color, lw=1.5, alpha=0.8
            )

    def _draw_knife(self, q: np.ndarray):
        """Draw simplified knife model at end-effector."""
        T_ee = self.kin.forward(q)
        origin = T_ee[:3, 3]
        R = T_ee[:3, :3]
        z_axis = R[:, 2]  # Knife axis (along Z of tool frame)
        y_axis = R[:, 1]  # Blade width direction

        # Handle (base to bolster)
        handle_end = origin + KNIFE_HANDLE_LEN * z_axis
        self.ax.plot(
            [origin[0], handle_end[0]],
            [origin[1], handle_end[1]],
            [origin[2], handle_end[2]],
            color=KNIFE_HANDLE_COLOR, lw=5, solid_capstyle="round"
        )

        # Blade (bolster to tip)
        blade_tip = handle_end + KNIFE_BLADE_LEN * z_axis
        self.ax.plot(
            [handle_end[0], blade_tip[0]],
            [handle_end[1], blade_tip[1]],
            [handle_end[2], blade_tip[2]],
            color=KNIFE_BLADE_COLOR, lw=2.5, solid_capstyle="round"
        )

        # Blade face (thin rectangle using poly3d)
        half_w = KNIFE_BLADE_WIDTH / 2
        corners = np.array([
            handle_end - half_w * y_axis,
            handle_end + half_w * y_axis,
            blade_tip + half_w * y_axis,
            blade_tip - half_w * y_axis,
        ])
        poly = Poly3DCollection(
            [corners], alpha=0.25, facecolor=KNIFE_BLADE_COLOR,
            edgecolor="#888888", linewidth=0.5
        )
        self.ax.add_collection3d(poly)

    def _draw_route(self):
        """Draw waypoints and route path."""
        if self._route is None or len(self._route) == 0:
            return

        positions = self._route.positions_array()  # (N, 3)

        # Route path line
        self.ax.plot(
            positions[:, 0], positions[:, 1], positions[:, 2],
            color=ROUTE_PATH_COLOR, lw=1.5, alpha=0.7, linestyle="-",
            zorder=3
        )

        # Waypoint markers
        for i, wp in enumerate(self._route.waypoints):
            is_selected = (i == self._selected_wp_idx)
            color = ACTIVE_WAYPOINT_COLOR if is_selected else WAYPOINT_COLOR
            size = 80 if is_selected else 40
            self.ax.scatter(
                [wp.x], [wp.y], [wp.z],
                c=color, s=size, zorder=7, depthshade=False,
                marker="o" if not is_selected else "*"
            )
            # Label
            if wp.label:
                self.ax.text(
                    wp.x + 10, wp.y + 10, wp.z + 10,
                    f"{i+1}:{wp.label}",
                    color="white", fontsize=6, alpha=0.8
                )
            else:
                self.ax.text(
                    wp.x + 10, wp.y + 10, wp.z + 10,
                    f"P[{i+1}]",
                    color="lightgray", fontsize=6, alpha=0.6
                )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy(self):
        """Clean up matplotlib resources."""
        plt.close(self.fig)
