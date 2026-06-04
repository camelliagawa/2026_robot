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
    from ..robot.tool_frame import ToolFrame
    from ..robot.user_frame import UserFrame


# Visual constants
ROBOT_COLOR = "#F5C400"       # FANUC yellow
ROBOT_JOINT_COLOR = "#333333"
KNIFE_BLADE_COLOR = "#C0C0C8"
KNIFE_HANDLE_COLOR = "#3A2010"
ROUTE_PATH_COLOR = "#2288FF"
WAYPOINT_COLOR = "#FF4422"
ACTIVE_WAYPOINT_COLOR = "#00FF88"
TCP_COLOR = "#00FFCC"         # Tool Center Point
USER_FRAME_COLOR = "#FF88FF"  # User frame axes
JOG_TARGET_COLOR = "#44FF44"  # Jog target crosshair
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
        self._tool_frame: Optional["ToolFrame"] = None
        self._user_frame: Optional["UserFrame"] = None
        self._jog_target: Optional[np.ndarray] = None  # (3,) position to show

        # Zoom state (1.0 = default view, smaller = zoomed in)
        self._zoom_scale: float = 1.0
        # View angle state
        self._elev: float = 20.0
        self._azim: float = -60.0

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

        # Mouse wheel zoom
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

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

    def set_tool_frame(self, tool_frame: Optional["ToolFrame"]):
        """Set the tool frame for TCP visualization."""
        self._tool_frame = tool_frame
        self._redraw()

    def set_user_frame(self, user_frame: Optional["UserFrame"]):
        """Set the user frame for axis visualization."""
        self._user_frame = user_frame
        self._redraw()

    def set_jog_target(self, position: Optional[np.ndarray]):
        """Set an IK jog target position to visualize (or None to hide)."""
        self._jog_target = position
        self._redraw()

    def refresh(self):
        """Force redraw."""
        self._redraw()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _on_scroll(self, event):
        """Zoom in/out with mouse wheel."""
        if event.button == "up":
            self._zoom_scale *= 0.85
        elif event.button == "down":
            self._zoom_scale *= 1.18
        self._zoom_scale = float(np.clip(self._zoom_scale, 0.05, 5.0))
        self._redraw()

    def _redraw(self):
        """Clear and redraw everything, preserving current view angle."""
        # Preserve view angle set by the user via mouse drag
        self._elev = float(self.ax.elev)
        self._azim = float(self.ax.azim)

        self.ax.cla()
        self._setup_axes()
        self.ax.view_init(elev=self._elev, azim=self._azim)
        self._draw_workspace_sphere()
        self._draw_user_frame()
        self._draw_robot(self._joint_angles)
        self._draw_route()
        self._draw_jog_target()
        self.canvas.draw_idle()

    def _setup_axes(self):
        """Configure 3D axes appearance."""
        ax = self.ax
        ax.set_facecolor("#1A1A1A")

        lim = 900 * self._zoom_scale
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(0, lim * 1.6)

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

        # Base platform disc
        theta = np.linspace(0, 2 * np.pi, 30)
        r = 120
        ax.plot(
            r * np.cos(theta), r * np.sin(theta), np.zeros(30),
            color="#666666", lw=2.0, alpha=0.6
        )
        # Ground plane grid lines
        for gv in np.linspace(-lim, lim, 9):
            ax.plot([gv, gv], [-lim, lim], [0, 0], color="#2A2A2A", lw=0.5, alpha=0.4)
            ax.plot([-lim, lim], [gv, gv], [0, 0], color="#2A2A2A", lw=0.5, alpha=0.4)

    def _draw_workspace_sphere(self):
        """Draw dashed circles indicating approximate workspace boundary."""
        reach = 911  # mm, max reach of LR Mate 200iD/14L
        theta = np.linspace(0, 2 * np.pi, 72)
        base_z = 330  # base height

        # Horizontal circle at shoulder height
        self.ax.plot(
            reach * np.cos(theta), reach * np.sin(theta),
            np.full(72, base_z),
            color="#334466", lw=0.8, alpha=0.35, linestyle="--"
        )
        # Vertical cross-section circle (XZ plane)
        self.ax.plot(
            reach * np.cos(theta), np.zeros(72),
            base_z + reach * np.sin(theta),
            color="#334466", lw=0.6, alpha=0.20, linestyle=":"
        )

    def _draw_robot(self, q: np.ndarray):
        """Draw robot links and joints."""
        positions = self.kin.get_joint_positions(q)  # (7, 3)

        xs = positions[:, 0]
        ys = positions[:, 1]
        zs = positions[:, 2]

        # Shadow on ground plane (Z=0)
        self.ax.plot(xs, ys, np.zeros_like(zs),
                     color="#333333", lw=2, alpha=0.4, linestyle="-")

        # Link segments — draw as thick yellow line (FANUC color)
        self.ax.plot(xs, ys, zs,
                     color=ROBOT_COLOR, lw=5, solid_capstyle="round",
                     zorder=5)

        # Joint markers
        joint_labels = ["Base", "J1", "J2", "J3", "J4", "J5", "J6"]
        for i, (x, y, z) in enumerate(positions):
            if i == 0:
                color, size = "#888888", 80   # base
            elif i == 6:
                color, size = "#FF8800", 60   # flange
            else:
                color, size = "#222222", 50
            self.ax.scatter([x], [y], [z], c=color, s=size, zorder=6, depthshade=False)

        # Draw knife at end-effector
        self._draw_knife(q)

        # EE coordinate frame (X=red, Y=green, Z=blue)
        T_ee = self.kin.forward(q)
        origin = T_ee[:3, 3]
        R = T_ee[:3, :3]
        scale = 80
        for col, color in enumerate(["red", "green", "blue"]):
            axis = origin + scale * R[:, col]
            self.ax.plot(
                [origin[0], axis[0]],
                [origin[1], axis[1]],
                [origin[2], axis[2]],
                color=color, lw=2.0, alpha=0.9
            )

        # Draw TCP (Tool Center Point) if tool frame is set
        self._draw_tcp(q, T_ee)

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

    def _draw_tcp(self, q: np.ndarray, T_ee: np.ndarray):
        """Draw TCP (Tool Center Point) after applying tool frame."""
        if self._tool_frame is None or self._tool_frame.z == 0.0:
            return
        T_tool = self._tool_frame.to_transform()
        T_tcp = T_ee @ T_tool
        tcp_pos = T_tcp[:3, 3]

        # Bright dot at TCP
        self.ax.scatter(
            [tcp_pos[0]], [tcp_pos[1]], [tcp_pos[2]],
            c=TCP_COLOR, s=100, zorder=8, depthshade=False,
            marker="*"
        )
        # Line from flange to TCP
        flange_pos = T_ee[:3, 3]
        self.ax.plot(
            [flange_pos[0], tcp_pos[0]],
            [flange_pos[1], tcp_pos[1]],
            [flange_pos[2], tcp_pos[2]],
            color=TCP_COLOR, lw=1.5, alpha=0.7, linestyle="--"
        )
        # TCP label
        self.ax.text(
            tcp_pos[0] + 10, tcp_pos[1] + 10, tcp_pos[2] + 10,
            "TCP", color=TCP_COLOR, fontsize=7, alpha=0.9
        )

    def _draw_user_frame(self):
        """Draw user frame coordinate axes at its position."""
        if self._user_frame is None:
            return
        T_uf = self._user_frame.to_transform()
        origin = T_uf[:3, 3]
        R = T_uf[:3, :3]
        scale = 120

        axis_colors = [USER_FRAME_COLOR, "#88FF88", "#8888FF"]
        axis_labels = ["Ux", "Uy", "Uz"]
        for col, (color, label) in enumerate(zip(axis_colors, axis_labels)):
            tip = origin + scale * R[:, col]
            self.ax.plot(
                [origin[0], tip[0]],
                [origin[1], tip[1]],
                [origin[2], tip[2]],
                color=color, lw=2.0, alpha=0.85
            )
            self.ax.text(
                tip[0], tip[1], tip[2],
                label, color=color, fontsize=7
            )

        # Origin marker
        self.ax.scatter(
            [origin[0]], [origin[1]], [origin[2]],
            c=USER_FRAME_COLOR, s=60, zorder=7, depthshade=False,
            marker="D"
        )
        name = getattr(self._user_frame, "name", "UF")
        self.ax.text(
            origin[0] + 15, origin[1] + 15, origin[2] + 15,
            f"[{name}]", color=USER_FRAME_COLOR, fontsize=7, alpha=0.85
        )

    def _draw_jog_target(self):
        """Draw jog target crosshair at the IK target position."""
        if self._jog_target is None:
            return
        x, y, z = self._jog_target
        s = 30  # crosshair half-size mm
        # Crosshair lines
        for dx, dy, dz in [(s, 0, 0), (0, s, 0), (0, 0, s)]:
            self.ax.plot(
                [x - dx, x + dx], [y - dy, y + dy], [z - dz, z + dz],
                color=JOG_TARGET_COLOR, lw=1.5, alpha=0.9
            )
        self.ax.scatter(
            [x], [y], [z], c=JOG_TARGET_COLOR, s=80,
            zorder=9, depthshade=False, marker="+"
        )
        self.ax.text(
            x + 12, y + 12, z + 12,
            f"({x:.0f},{y:.0f},{z:.0f})",
            color=JOG_TARGET_COLOR, fontsize=6, alpha=0.85
        )

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
