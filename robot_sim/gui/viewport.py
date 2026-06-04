"""
3D Viewport for the robot simulation using matplotlib.

Renders the FANUC LR Mate 200iD/14L with realistic cylindrical arm segments,
route waypoints, user frame axes, TCP marker, and workspace boundary.
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

# ── カラー定数 ─────────────────────────────────────────────────────────
FANUC_YELLOW    = "#F5C400"   # FANUC ロボット本体色
FANUC_YELLOW_D  = "#C49A00"   # 暗面
FANUC_BLACK     = "#1A1A1A"   # 関節部
KNIFE_BLADE     = "#C8C8D0"
KNIFE_HANDLE    = "#3A2010"
ROUTE_COLOR     = "#2288FF"
WP_COLOR        = "#FF4422"
WP_ACTIVE       = "#00FF88"
TCP_COLOR       = "#00FFCC"
UFRAME_COLOR    = "#FF88FF"
JOG_COLOR       = "#44FF44"
GRID_ALPHA      = 0.25

KNIFE_HANDLE_LEN  = 150.0
KNIFE_BLADE_LEN   = 200.0
KNIFE_BLADE_WIDTH = 45.0


# ── 3D プリミティブ描画ヘルパー ─────────────────────────────────────────

def _cylinder(ax, p1, p2, radius: float, color: str,
              alpha: float = 1.0, n: int = 10, shade: bool = True):
    """Draw a cylinder from p1 to p2."""
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    v  = p2 - p1
    ln = np.linalg.norm(v)
    if ln < 1e-6:
        return
    v_u = v / ln

    # Two perpendicular axes
    ref = [1, 0, 0] if abs(v_u[0]) < 0.9 else [0, 1, 0]
    e1  = np.cross(v_u, ref); e1 /= np.linalg.norm(e1)
    e2  = np.cross(v_u, e1)

    theta = np.linspace(0, 2 * np.pi, n + 1)
    X = np.zeros((2, n + 1))
    Y = np.zeros((2, n + 1))
    Z = np.zeros((2, n + 1))
    for j, t in enumerate(theta):
        d = radius * (np.cos(t) * e1 + np.sin(t) * e2)
        for row, base in enumerate([p1, p2]):
            X[row, j] = base[0] + d[0]
            Y[row, j] = base[1] + d[1]
            Z[row, j] = base[2] + d[2]
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha,
                    shade=shade, linewidth=0, antialiased=False)


def _sphere(ax, center, radius: float, color: str,
            alpha: float = 1.0, n: int = 8):
    """Draw a sphere."""
    c = np.asarray(center, float)
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = c[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = c[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = c[2] + radius * np.outer(np.ones(n), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha,
                    shade=True, linewidth=0, antialiased=False)


def _disk(ax, center, normal, radius: float, color: str,
          alpha: float = 1.0, n: int = 16):
    """Draw a filled disk (end cap)."""
    c  = np.asarray(center, float)
    nv = np.asarray(normal, float)
    ln = np.linalg.norm(nv)
    if ln < 1e-6:
        return
    nv /= ln
    ref = [1, 0, 0] if abs(nv[0]) < 0.9 else [0, 1, 0]
    e1  = np.cross(nv, ref)
    if np.linalg.norm(e1) < 1e-6:
        return
    e1 /= np.linalg.norm(e1)
    e2  = np.cross(nv, e1)
    theta = np.linspace(0, 2 * np.pi, n)
    verts = [c + radius * (np.cos(t) * e1 + np.sin(t) * e2) for t in theta]
    poly  = Poly3DCollection([verts], alpha=alpha,
                              facecolor=color, edgecolor="none")
    ax.add_collection3d(poly)


class Viewport3D:
    """Embedded 3D matplotlib viewport inside a tkinter frame."""

    def __init__(self, parent: tk.Widget, kinematics: "Kinematics"):
        self.kin = kinematics
        self._route: Optional["Route"]      = None
        self._selected_wp_idx: Optional[int] = None
        self._joint_angles                   = np.zeros(6)
        self._tool_frame: Optional["ToolFrame"] = None
        self._user_frame: Optional["UserFrame"] = None
        self._jog_target: Optional[np.ndarray]  = None

        self._zoom_scale: float = 1.0
        self._elev: float = 22.0
        self._azim: float = -55.0

        self.fig = plt.figure(figsize=(7, 6), facecolor="#161B22")
        self.ax: Axes3D = self.fig.add_subplot(111, projection="3d")
        self._setup_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        toolbar_frame = tk.Frame(parent, bg="#161B22")
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.update_robot(self._joint_angles)

    # ── Public interface ───────────────────────────────────────────────

    def update_robot(self, joint_angles: np.ndarray):
        self._joint_angles = np.asarray(joint_angles)
        self._redraw()

    def set_route(self, route: Optional["Route"]):
        self._route = route
        self._redraw()

    def set_selected_waypoint(self, idx: Optional[int]):
        self._selected_wp_idx = idx
        self._redraw()

    def set_tool_frame(self, tool_frame: Optional["ToolFrame"]):
        self._tool_frame = tool_frame
        self._redraw()

    def set_user_frame(self, user_frame: Optional["UserFrame"]):
        self._user_frame = user_frame
        self._redraw()

    def set_jog_target(self, position: Optional[np.ndarray]):
        self._jog_target = position
        self._redraw()

    def refresh(self):
        self._redraw()

    # ── Drawing ────────────────────────────────────────────────────────

    def _on_scroll(self, event):
        if event.button == "up":
            self._zoom_scale *= 0.85
        elif event.button == "down":
            self._zoom_scale *= 1.18
        self._zoom_scale = float(np.clip(self._zoom_scale, 0.05, 5.0))
        self._redraw()

    def _redraw(self):
        self._elev = float(self.ax.elev)
        self._azim = float(self.ax.azim)
        self.ax.cla()
        self._setup_axes()
        self.ax.view_init(elev=self._elev, azim=self._azim)
        self._draw_workspace()
        self._draw_user_frame()
        self._draw_robot(self._joint_angles)
        self._draw_route()
        self._draw_jog_target()
        self.canvas.draw_idle()

    def _setup_axes(self):
        ax = self.ax
        ax.set_facecolor("#0D1117")

        lim = 900 * self._zoom_scale
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(0, lim * 1.6)

        ax.set_xlabel("X (mm)", color="#8B949E", fontsize=8, labelpad=6)
        ax.set_ylabel("Y (mm)", color="#8B949E", fontsize=8, labelpad=6)
        ax.set_zlabel("Z (mm)", color="#8B949E", fontsize=8, labelpad=6)
        ax.tick_params(colors="#444C56", labelsize=7)

        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#21262D")
        ax.grid(True, alpha=0.15, color="#444C56")

        # Ground grid
        for gv in np.linspace(-lim, lim, 7):
            ax.plot([gv, gv], [-lim, lim], [0, 0],
                    color="#21262D", lw=0.5, alpha=0.5)
            ax.plot([-lim, lim], [gv, gv], [0, 0],
                    color="#21262D", lw=0.5, alpha=0.5)

    def _draw_workspace(self):
        """Workspace boundary circles."""
        reach = 911
        theta = np.linspace(0, 2 * np.pi, 72)
        base_z = 330

        self.ax.plot(reach * np.cos(theta), reach * np.sin(theta),
                     np.full(72, base_z),
                     color="#1E3A5F", lw=1.0, alpha=0.5, linestyle="--")
        self.ax.plot(reach * np.cos(theta), np.zeros(72),
                     base_z + reach * np.sin(theta),
                     color="#1E3A5F", lw=0.6, alpha=0.3, linestyle=":")
        # Label
        self.ax.text(reach * 0.72, 0, base_z + reach * 0.72,
                     "911mm", color="#1E5A8F", fontsize=6, alpha=0.6)

    def _draw_robot(self, q: np.ndarray):
        """Draw FANUC LR Mate 200iD/14L with realistic cylindrical geometry."""
        pos = self.kin.get_joint_positions(q)  # (7, 3)  Base + J1…J6

        # Link radii (mm) — proportional to actual robot geometry
        radii = [80, 55, 48, 38, 32, 24, 20]

        # ── ベース（土台） ────────────────────────────────────────────
        base    = pos[0].copy()
        j1_pos  = pos[1].copy()

        # 底板（黒い正方形台）
        hw = 110
        corners = np.array([
            [-hw, -hw, 0], [hw, -hw, 0],
            [hw,  hw, 0],  [-hw,  hw, 0],
        ], float) + base
        base_top_corners = corners.copy(); base_top_corners[:, 2] += 60
        for a, b in zip(corners, base_top_corners):
            self.ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                         color="#333333", lw=1.5)
        for loop in [corners, base_top_corners]:
            lp = np.vstack([loop, loop[0]])
            self.ax.plot(lp[:, 0], lp[:, 1], lp[:, 2],
                         color="#333333", lw=1.5)

        # J1回転胴（円柱）— ベース上面〜J1位置
        base_top = base.copy(); base_top[2] = 60
        _cylinder(self.ax, base_top, j1_pos, 75, FANUC_BLACK, alpha=0.95, n=12)
        _disk(self.ax, base_top, [0, 0, -1], 75, "#252525", alpha=0.9)
        _disk(self.ax, j1_pos,  [0, 0,  1], 75, "#252525", alpha=0.9)

        # ── 上腕（J1→J2）── 肩関節部 ───────────────────────────────
        j2_pos = pos[2].copy()

        # 肩球体
        _sphere(self.ax, j1_pos, 55, FANUC_BLACK, alpha=0.9, n=10)

        # 上腕リンク（太い黄色円柱）
        _cylinder(self.ax, j1_pos, j2_pos, 48, FANUC_YELLOW, alpha=1.0, n=12)
        _disk(self.ax, j1_pos, -(j2_pos - j1_pos), 48, FANUC_YELLOW_D, alpha=0.9)
        _disk(self.ax, j2_pos,   j2_pos - j1_pos,  48, FANUC_YELLOW_D, alpha=0.9)

        # ── 前腕（J2→J3）── 肘関節部 ────────────────────────────────
        j3_pos = pos[3].copy()

        # 肘球体
        _sphere(self.ax, j2_pos, 48, FANUC_BLACK, alpha=0.9, n=10)

        # 前腕リンク
        _cylinder(self.ax, j2_pos, j3_pos, 40, FANUC_YELLOW, alpha=1.0, n=12)
        _disk(self.ax, j2_pos, -(j3_pos - j2_pos), 40, FANUC_YELLOW_D, alpha=0.9)
        _disk(self.ax, j3_pos,   j3_pos - j2_pos,  40, FANUC_YELLOW_D, alpha=0.9)

        # ── 手首部（J3→J4）─────────────────────────────────────────
        j4_pos = pos[4].copy()

        _sphere(self.ax, j3_pos, 40, FANUC_BLACK, alpha=0.9, n=10)
        _cylinder(self.ax, j3_pos, j4_pos, 32, FANUC_YELLOW, alpha=1.0, n=10)
        _disk(self.ax, j3_pos, -(j4_pos - j3_pos), 32, FANUC_YELLOW_D)
        _disk(self.ax, j4_pos,   j4_pos - j3_pos,  32, FANUC_YELLOW_D)

        # ── 手首ピッチ（J4→J5）────────────────────────────────────
        j5_pos = pos[5].copy()

        _sphere(self.ax, j4_pos, 32, FANUC_BLACK, alpha=0.9, n=8)
        _cylinder(self.ax, j4_pos, j5_pos, 26, FANUC_YELLOW, alpha=1.0, n=10)
        _disk(self.ax, j4_pos, -(j5_pos - j4_pos), 26, FANUC_YELLOW_D)
        _disk(self.ax, j5_pos,   j5_pos - j4_pos,  26, FANUC_YELLOW_D)

        # ── フランジ（J5→J6）──────────────────────────────────────
        j6_pos = pos[6].copy()

        _sphere(self.ax, j5_pos, 26, FANUC_BLACK, alpha=0.9, n=8)
        _cylinder(self.ax, j5_pos, j6_pos, 22, FANUC_YELLOW, alpha=1.0, n=10)

        # フランジ板（黒い円盤）
        ee_dir = j6_pos - j5_pos
        if np.linalg.norm(ee_dir) > 1e-3:
            _disk(self.ax, j6_pos, ee_dir, 32, FANUC_BLACK, alpha=0.95)
            _cylinder(self.ax, j6_pos, j6_pos + ee_dir / np.linalg.norm(ee_dir) * 15,
                      32, "#222222", alpha=0.9, n=10)

        # ── 地面の影 ─────────────────────────────────────────────────
        shadow_xs = pos[:, 0]; shadow_ys = pos[:, 1]
        self.ax.plot(shadow_xs, shadow_ys, np.zeros(len(pos)),
                     color="#333333", lw=3, alpha=0.25, linestyle="-")

        # ── EE 座標フレーム ───────────────────────────────────────────
        T_ee = self.kin.forward(q)
        origin = T_ee[:3, 3]
        R = T_ee[:3, :3]
        scale = 70
        axis_colors = ["#FF4444", "#44FF44", "#4444FF"]
        axis_names  = ["X", "Y", "Z"]
        for col, (color, name) in enumerate(zip(axis_colors, axis_names)):
            tip = origin + scale * R[:, col]
            self.ax.plot([origin[0], tip[0]], [origin[1], tip[1]],
                         [origin[2], tip[2]], color=color, lw=2.0, alpha=0.9)
            self.ax.text(tip[0], tip[1], tip[2], name,
                         color=color, fontsize=6, alpha=0.85)

        # ── 包丁・TCP ────────────────────────────────────────────────
        self._draw_knife(q, T_ee)
        self._draw_tcp(q, T_ee)

    def _draw_knife(self, q: np.ndarray, T_ee: np.ndarray):
        """Draw simplified knife model at end-effector."""
        origin = T_ee[:3, 3]
        R = T_ee[:3, :3]
        z_axis = R[:, 2]
        y_axis = R[:, 1]

        # ハンドル
        handle_end = origin + KNIFE_HANDLE_LEN * z_axis
        self.ax.plot([origin[0], handle_end[0]],
                     [origin[1], handle_end[1]],
                     [origin[2], handle_end[2]],
                     color=KNIFE_HANDLE, lw=5, solid_capstyle="round")

        # 刃
        blade_tip = handle_end + KNIFE_BLADE_LEN * z_axis
        self.ax.plot([handle_end[0], blade_tip[0]],
                     [handle_end[1], blade_tip[1]],
                     [handle_end[2], blade_tip[2]],
                     color=KNIFE_BLADE, lw=2.5, solid_capstyle="round")

        # 刃面（半透明ポリゴン）
        hw = KNIFE_BLADE_WIDTH / 2
        corners = np.array([
            handle_end - hw * y_axis,
            handle_end + hw * y_axis,
            blade_tip  + hw * y_axis,
            blade_tip  - hw * y_axis,
        ])
        poly = Poly3DCollection([corners], alpha=0.22,
                                facecolor=KNIFE_BLADE,
                                edgecolor="#666666", linewidth=0.5)
        self.ax.add_collection3d(poly)

    def _draw_tcp(self, q: np.ndarray, T_ee: np.ndarray):
        """Draw TCP marker if tool frame has offset."""
        if self._tool_frame is None or self._tool_frame.z == 0.0:
            return
        T_tcp   = T_ee @ self._tool_frame.to_transform()
        tcp_pos = T_tcp[:3, 3]
        flange  = T_ee[:3, 3]

        self.ax.scatter([tcp_pos[0]], [tcp_pos[1]], [tcp_pos[2]],
                        c=TCP_COLOR, s=120, zorder=8,
                        depthshade=False, marker="*")
        self.ax.plot([flange[0], tcp_pos[0]],
                     [flange[1], tcp_pos[1]],
                     [flange[2], tcp_pos[2]],
                     color=TCP_COLOR, lw=1.5, alpha=0.7, linestyle="--")
        self.ax.text(tcp_pos[0] + 12, tcp_pos[1] + 12, tcp_pos[2] + 12,
                     "TCP", color=TCP_COLOR, fontsize=7, alpha=0.9)

    def _draw_user_frame(self):
        """Draw user frame coordinate axes."""
        if self._user_frame is None:
            return
        T_uf   = self._user_frame.to_transform()
        origin = T_uf[:3, 3]
        R      = T_uf[:3, :3]
        scale  = 120

        colors = [UFRAME_COLOR, "#88FF88", "#8888FF"]
        labels = ["Ux", "Uy", "Uz"]
        for col, (color, lbl) in enumerate(zip(colors, labels)):
            tip = origin + scale * R[:, col]
            self.ax.plot([origin[0], tip[0]], [origin[1], tip[1]],
                         [origin[2], tip[2]], color=color, lw=2.0, alpha=0.85)
            self.ax.text(tip[0], tip[1], tip[2],
                         lbl, color=color, fontsize=7)

        self.ax.scatter([origin[0]], [origin[1]], [origin[2]],
                        c=UFRAME_COLOR, s=60, zorder=7,
                        depthshade=False, marker="D")
        name = getattr(self._user_frame, "name", "UF")
        self.ax.text(origin[0] + 15, origin[1] + 15, origin[2] + 15,
                     f"[{name}]", color=UFRAME_COLOR, fontsize=7, alpha=0.85)

    def _draw_jog_target(self):
        """Draw jog target crosshair."""
        if self._jog_target is None:
            return
        x, y, z = self._jog_target
        s = 30
        for dx, dy, dz in [(s, 0, 0), (0, s, 0), (0, 0, s)]:
            self.ax.plot([x - dx, x + dx], [y - dy, y + dy],
                         [z - dz, z + dz],
                         color=JOG_COLOR, lw=1.5, alpha=0.9)
        self.ax.scatter([x], [y], [z], c=JOG_COLOR, s=80,
                        zorder=9, depthshade=False, marker="+")
        self.ax.text(x + 12, y + 12, z + 12,
                     f"({x:.0f},{y:.0f},{z:.0f})",
                     color=JOG_COLOR, fontsize=6, alpha=0.85)

    def _draw_route(self):
        """Draw waypoints and route path."""
        if self._route is None or len(self._route) == 0:
            return

        positions = self._route.positions_array()

        self.ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                     color=ROUTE_COLOR, lw=1.5, alpha=0.7, zorder=3)

        for i, wp in enumerate(self._route.waypoints):
            selected = (i == self._selected_wp_idx)
            color    = WP_ACTIVE if selected else WP_COLOR
            size     = 100 if selected else 45
            marker   = "*" if selected else "o"
            self.ax.scatter([wp.x], [wp.y], [wp.z],
                            c=color, s=size, zorder=7,
                            depthshade=False, marker=marker)
            label_text = f"{i+1}:{wp.label}" if wp.label else f"P[{i+1}]"
            fg = "white" if selected else "#AAAAAA"
            self.ax.text(wp.x + 10, wp.y + 10, wp.z + 10,
                         label_text, color=fg, fontsize=6, alpha=0.85)

    # ── Cleanup ────────────────────────────────────────────────────────

    def destroy(self):
        plt.close(self.fig)
