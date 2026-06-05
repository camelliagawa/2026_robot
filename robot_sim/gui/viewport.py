"""
3D Viewport for the robot simulation using matplotlib.

Renders the FANUC LR Mate 200iD/14L with realistic geometry,
route waypoints, user frame axes, TCP marker, and workspace boundary.
"""
from __future__ import annotations

import os
from typing import Optional, List, TYPE_CHECKING

import numpy as np

try:
    from stl import mesh as _stl_mesh
    _HAS_STL = True
except ImportError:
    _HAS_STL = False
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
FANUC_DARK_GRAY = "#2E2E2E"   # ベース台座
KNIFE_BLADE     = "#C8C8D0"
KNIFE_HANDLE    = "#3A2010"
ROUTE_COLOR     = "#2288FF"
WP_COLOR        = "#FF4422"
WP_ACTIVE       = "#00FF88"
TCP_COLOR       = "#00FFCC"
UFRAME_COLOR    = "#FF88FF"
JOG_COLOR       = "#44FF44"

KNIFE_HANDLE_LEN  = 150.0
KNIFE_BLADE_LEN   = 200.0
KNIFE_BLADE_WIDTH = 45.0


# ── 3D プリミティブ描画ヘルパー ─────────────────────────────────────────

def _cylinder(ax, p1, p2, radius: float, color: str,
              alpha: float = 1.0, n: int = 10):
    """Draw a cylinder from p1 to p2."""
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    v  = p2 - p1
    ln = np.linalg.norm(v)
    if ln < 1e-6:
        return
    v_u = v / ln
    ref = [1, 0, 0] if abs(v_u[0]) < 0.9 else [0, 1, 0]
    e1  = np.cross(v_u, ref); e1 /= np.linalg.norm(e1)
    e2  = np.cross(v_u, e1)
    theta = np.linspace(0, 2 * np.pi, n + 1)
    X = np.zeros((2, n + 1)); Y = np.zeros((2, n + 1)); Z = np.zeros((2, n + 1))
    for j, t in enumerate(theta):
        d = radius * (np.cos(t) * e1 + np.sin(t) * e2)
        for row, base in enumerate([p1, p2]):
            X[row, j] = base[0] + d[0]
            Y[row, j] = base[1] + d[1]
            Z[row, j] = base[2] + d[2]
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha,
                    shade=True, linewidth=0, antialiased=False)


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


def _box_link(ax, p1, p2, w: float, h: float, color: str, alpha: float = 1.0):
    """
    Draw a rectangular-section link from p1 to p2.
    w = width, h = height of the cross-section (perpendicular to link axis).
    Gives a more realistic arm appearance than a cylinder.
    """
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    v  = p2 - p1
    ln = np.linalg.norm(v)
    if ln < 1e-6:
        return
    v_u = v / ln
    ref = [0, 0, 1] if abs(v_u[2]) < 0.9 else [1, 0, 0]
    e1  = np.cross(v_u, ref); e1 /= np.linalg.norm(e1)
    e2  = np.cross(v_u, e1)

    hw, hh = w / 2, h / 2
    # 8 corners: 4 at p1 end, 4 at p2 end
    offsets = [hw * e1 + hh * e2, -hw * e1 + hh * e2,
               -hw * e1 - hh * e2,  hw * e1 - hh * e2]
    c = [p1 + o for o in offsets] + [p2 + o for o in offsets]

    edge_c = FANUC_YELLOW_D if color == FANUC_YELLOW else "#444444"
    faces = [
        [c[0], c[1], c[2], c[3]],  # p1 cap
        [c[4], c[5], c[6], c[7]],  # p2 cap
        [c[0], c[1], c[5], c[4]],  # side A
        [c[2], c[3], c[7], c[6]],  # side B
        [c[1], c[2], c[6], c[5]],  # side C
        [c[0], c[3], c[7], c[4]],  # side D
    ]
    poly = Poly3DCollection(faces, alpha=alpha, facecolor=color,
                            edgecolor=edge_c, linewidth=0.4)
    ax.add_collection3d(poly)


def _rotated_box(ax, center, R: np.ndarray,
                 lx: float, ly: float, lz: float,
                 color: str, alpha: float = 1.0):
    """
    Draw a box rotated by matrix R, centered at `center`.
    lx/ly/lz are full side lengths along R's x/y/z columns.
    """
    c = np.asarray(center, float)
    signs = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                       [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1]], float)
    half  = np.array([lx/2, ly/2, lz/2])
    verts = c + (R @ (signs * half).T).T

    edge_c = FANUC_YELLOW_D if color == FANUC_YELLOW else "#444444"
    v = verts
    faces = [
        [v[0],v[1],v[2],v[3]], [v[4],v[5],v[6],v[7]],
        [v[0],v[1],v[5],v[4]], [v[2],v[3],v[7],v[6]],
        [v[1],v[2],v[6],v[5]], [v[0],v[3],v[7],v[4]],
    ]
    poly = Poly3DCollection(faces, alpha=alpha, facecolor=color,
                            edgecolor=edge_c, linewidth=0.4)
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
        self._elev: float = 25.0
        self._azim: float = -45.0

        self._stl_verts: Optional[np.ndarray] = None   # (N,3,3) STL triangles
        self._stl_name: str = ""
        self._stl_T: np.ndarray = np.eye(4)
        self._csv_points: Optional[np.ndarray] = None  # (N,3) CSV points
        self._csv_name: str = ""
        self._csv_T: np.ndarray = np.eye(4)

        self._tcp_markers: List[dict] = []    # [{"name": str, "pos": np.ndarray}]
        self._target_markers: List[dict] = [] # [{"name": str, "pos": np.ndarray}]

        self.fig = plt.figure(facecolor="#161B22")
        self.fig.subplots_adjust(left=-0.08, right=1.08, bottom=-0.08, top=1.08)
        self.ax: Axes3D = self.fig.add_subplot(111, projection="3d")
        self._setup_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Disable built-in 3D left-drag rotation
        for cid in list(getattr(self.ax, '_cids', [])):
            self.canvas.mpl_disconnect(cid)
        if hasattr(self.ax, '_cids'):
            self.ax._cids.clear()

        for ev in ("button_press_event", "button_release_event", "motion_notify_event"):
            self.canvas.mpl_connect(ev, lambda e: None)

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
        self._draw_overlay()
        self._draw_markers()
        self._draw_route()
        self._draw_jog_target()
        self.canvas.draw_idle()

    def _setup_axes(self):
        ax = self.ax
        ax.set_facecolor("#0D1117")

        lim = 700 * self._zoom_scale
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(0, lim * 1.6)

        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False,
                       colors="#444C56", labelsize=0, length=0)

        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#21262D")
        ax.grid(False)

    def _draw_workspace(self):
        """Workspace boundary circle (horizontal plane at shoulder height)."""
        reach = self.kin.dh.REACH_MM
        theta = np.linspace(0, 2 * np.pi, 72)
        base_z = self.kin.dh.joints[0].d  # d1 = 330

        self.ax.plot(reach * np.cos(theta), reach * np.sin(theta),
                     np.full(72, base_z),
                     color="#1E3A5F", lw=1.0, alpha=0.5, linestyle="--")
        self.ax.text(reach * 0.72, 0, base_z + 30,
                     f"{int(reach)}mm", color="#1E5A8F", fontsize=6, alpha=0.6)

    def _draw_robot(self, q: np.ndarray):
        """Draw FANUC LR Mate 200iD/14L with realistic cylindrical geometry."""
        pos = self.kin.get_joint_positions(q)  # (7, 3)  Base + J1…J6

        # ── ベース（土台） ────────────────────────────────────────────
        base   = pos[0].copy()
        j1_pos = pos[1].copy()

        # 底板（黒いワイヤーフレーム台）
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

        # J1 回転胴（黒円柱）
        base_top = base.copy(); base_top[2] = 60
        _cylinder(self.ax, base_top, j1_pos, 75, FANUC_BLACK, alpha=0.95, n=12)
        _disk(self.ax, base_top, [0, 0, -1], 75, "#252525", alpha=0.9)
        _disk(self.ax, j1_pos,  [0, 0,  1], 75, "#252525", alpha=0.9)

        # ── 上腕（J1→J2） ─────────────────────────────────────────
        j2_pos = pos[2].copy()
        _sphere(self.ax, j1_pos, 55, FANUC_BLACK, alpha=0.9, n=10)
        _cylinder(self.ax, j1_pos, j2_pos, 48, FANUC_YELLOW, alpha=1.0, n=12)
        _disk(self.ax, j1_pos, -(j2_pos - j1_pos), 48, FANUC_YELLOW_D, alpha=0.9)
        _disk(self.ax, j2_pos,   j2_pos - j1_pos,  48, FANUC_YELLOW_D, alpha=0.9)

        # ── 前腕（J2→J3） ─────────────────────────────────────────
        j3_pos = pos[3].copy()
        _sphere(self.ax, j2_pos, 48, FANUC_BLACK, alpha=0.9, n=10)
        _cylinder(self.ax, j2_pos, j3_pos, 40, FANUC_YELLOW, alpha=1.0, n=12)
        _disk(self.ax, j2_pos, -(j3_pos - j2_pos), 40, FANUC_YELLOW_D, alpha=0.9)
        _disk(self.ax, j3_pos,   j3_pos - j2_pos,  40, FANUC_YELLOW_D, alpha=0.9)

        # ── 手首部（J3→J4） ───────────────────────────────────────
        j4_pos = pos[4].copy()
        _sphere(self.ax, j3_pos, 40, FANUC_BLACK, alpha=0.9, n=10)
        _cylinder(self.ax, j3_pos, j4_pos, 32, FANUC_YELLOW, alpha=1.0, n=10)
        _disk(self.ax, j3_pos, -(j4_pos - j3_pos), 32, FANUC_YELLOW_D)
        _disk(self.ax, j4_pos,   j4_pos - j3_pos,  32, FANUC_YELLOW_D)

        # ── 手首ピッチ（J4→J5） ───────────────────────────────────
        j5_pos = pos[5].copy()
        _sphere(self.ax, j4_pos, 32, FANUC_BLACK, alpha=0.9, n=8)
        _cylinder(self.ax, j4_pos, j5_pos, 26, FANUC_YELLOW, alpha=1.0, n=10)
        _disk(self.ax, j4_pos, -(j5_pos - j4_pos), 26, FANUC_YELLOW_D)
        _disk(self.ax, j5_pos,   j5_pos - j4_pos,  26, FANUC_YELLOW_D)

        # ── フランジ（J5→J6） ─────────────────────────────────────
        j6_pos = pos[6].copy()
        _sphere(self.ax, j5_pos, 26, FANUC_BLACK, alpha=0.9, n=8)
        _cylinder(self.ax, j5_pos, j6_pos, 22, FANUC_YELLOW, alpha=1.0, n=10)
        ee_dir = j6_pos - j5_pos
        if np.linalg.norm(ee_dir) > 1e-3:
            _disk(self.ax, j6_pos, ee_dir, 32, FANUC_BLACK, alpha=0.95)
            _cylinder(self.ax, j6_pos,
                      j6_pos + ee_dir / np.linalg.norm(ee_dir) * 15,
                      32, "#222222", alpha=0.9, n=10)

        # ── 地面の影 ──────────────────────────────────────────────
        self.ax.plot(pos[:, 0], pos[:, 1], np.zeros(len(pos)),
                     color="#333333", lw=3, alpha=0.25)

        # ── EE 座標フレーム ────────────────────────────────────────
        T_ee = self.kin.forward(q)
        origin = T_ee[:3, 3]
        R = T_ee[:3, :3]
        for col, (color, name) in enumerate(
                zip(["#FF4444", "#44FF44", "#4444FF"], ["X", "Y", "Z"])):
            tip = origin + 70 * R[:, col]
            self.ax.plot([origin[0], tip[0]], [origin[1], tip[1]],
                         [origin[2], tip[2]], color=color, lw=2.0, alpha=0.9)
            self.ax.text(tip[0], tip[1], tip[2], name,
                         color=color, fontsize=6, alpha=0.85)

        self._draw_knife(q, T_ee)
        self._draw_tcp(q, T_ee)

    def _draw_knife(self, q: np.ndarray, T_ee: np.ndarray):
        """Draw simplified knife model at end-effector."""
        origin = T_ee[:3, 3]
        R = T_ee[:3, :3]
        z_axis = R[:, 2]
        y_axis = R[:, 1]

        handle_end = origin + KNIFE_HANDLE_LEN * z_axis
        self.ax.plot([origin[0], handle_end[0]],
                     [origin[1], handle_end[1]],
                     [origin[2], handle_end[2]],
                     color=KNIFE_HANDLE, lw=5, solid_capstyle="round")

        blade_tip = handle_end + KNIFE_BLADE_LEN * z_axis
        self.ax.plot([handle_end[0], blade_tip[0]],
                     [handle_end[1], blade_tip[1]],
                     [handle_end[2], blade_tip[2]],
                     color=KNIFE_BLADE, lw=2.5, solid_capstyle="round")

        hw = KNIFE_BLADE_WIDTH / 2
        corners = np.array([
            handle_end - hw * y_axis, handle_end + hw * y_axis,
            blade_tip  + hw * y_axis, blade_tip  - hw * y_axis,
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

        for col, (color, lbl) in enumerate(
                zip([UFRAME_COLOR, "#88FF88", "#8888FF"], ["Ux", "Uy", "Uz"])):
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

    # ── Overlay ────────────────────────────────────────────────────────

    def load_stl(self, path: str):
        if not _HAS_STL:
            return False
        m = _stl_mesh.Mesh.from_file(path)
        self._stl_verts = m.vectors.copy()  # (N,3,3)
        self._stl_name = os.path.basename(path)
        self._redraw()
        return True

    def load_csv_points(self, path: str):
        import csv
        pts = []
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    try:
                        pts.append([float(row[0]), float(row[1]), float(row[2])])
                    except ValueError:
                        pass
        if pts:
            self._csv_points = np.array(pts)
            self._csv_name = os.path.basename(path)
            self._redraw()
            return True
        return False

    def stl_bbox(self):
        """Return (xmin,xmax, ymin,ymax, zmin,zmax) of STL, or None."""
        if self._stl_verts is None:
            return None
        v = self._stl_verts.reshape(-1, 3)
        return (v[:,0].min(), v[:,0].max(),
                v[:,1].min(), v[:,1].max(),
                v[:,2].min(), v[:,2].max())

    def set_stl_pose(self, x, y, z, rx, ry, rz):
        from ..robot.kinematics import Kinematics
        self._stl_T = Kinematics.pose_to_transform(x, y, z, rx, ry, rz)
        self._redraw()

    def set_csv_pose(self, x, y, z, rx, ry, rz):
        from ..robot.kinematics import Kinematics
        self._csv_T = Kinematics.pose_to_transform(x, y, z, rx, ry, rz)
        self._redraw()

    def set_overlay_pose(self, x, y, z, rx, ry, rz):
        """Legacy: applies to whichever overlay is loaded (STL priority)."""
        if self._stl_verts is not None:
            self.set_stl_pose(x, y, z, rx, ry, rz)
        else:
            self.set_csv_pose(x, y, z, rx, ry, rz)

    def clear_stl(self):
        self._stl_verts = None
        self._stl_name = ""
        self._stl_T = np.eye(4)
        self._redraw()

    def clear_csv(self):
        self._csv_points = None
        self._csv_name = ""
        self._csv_T = np.eye(4)
        self._redraw()

    def clear_overlay(self):
        self.clear_stl()
        self.clear_csv()

    def _draw_overlay(self):
        if self._stl_verts is not None:
            R, t = self._stl_T[:3, :3], self._stl_T[:3, 3]
            all_verts = self._stl_verts.reshape(-1, 3)
            tv = ((R @ all_verts.T).T + t)
            tverts = tv.reshape(-1, 3, 3)
            # 頂点 scatter（確実に何か表示される）
            self.ax.scatter(tv[::5, 0], tv[::5, 1], tv[::5, 2],
                            c="#6699FF", s=2, alpha=0.5, depthshade=False)
            # ワイヤーフレーム（間引き）
            for tri in tverts[::4]:
                xs = [tri[0,0], tri[1,0], tri[2,0], tri[0,0]]
                ys = [tri[0,1], tri[1,1], tri[2,1], tri[0,1]]
                zs = [tri[0,2], tri[1,2], tri[2,2], tri[0,2]]
                self.ax.plot(xs, ys, zs, color="#6699FF", linewidth=0.4, alpha=0.5)
            ctr = tv.mean(axis=0)
            self.ax.text(ctr[0], ctr[1], ctr[2],
                         self._stl_name, color="#99BBFF", fontsize=7)
        if self._csv_points is not None:
            R, t = self._csv_T[:3, :3], self._csv_T[:3, 3]
            pts = (R @ self._csv_points.T).T + t
            self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                            c="#FF9944", s=8, alpha=0.6, depthshade=False)
            ctr = pts.mean(axis=0)
            self.ax.text(ctr[0], ctr[1], ctr[2],
                         self._csv_name, color="#FFBB66", fontsize=6)

    # ── Markers ────────────────────────────────────────────────────────

    def set_markers(self, tcp_markers: list, target_markers: list):
        """Replace all TCP/target markers. Each item: {"name": str, "pos": array-like}."""
        self._tcp_markers = [
            {"name": m["name"], "pos": np.asarray(m["pos"], float)} for m in tcp_markers
        ]
        self._target_markers = [
            {"name": m["name"], "pos": np.asarray(m["pos"], float)} for m in target_markers
        ]
        self._redraw()

    def _draw_markers(self):
        for m in self._tcp_markers:
            x, y, z = m["pos"]
            self.ax.scatter([x], [y], [z], c="#00FFCC", s=200, zorder=8,
                            depthshade=False, marker="*")
            self.ax.text(x + 14, y + 14, z + 14,
                         f"[TCP] {m['name']}", color="#00FFCC", fontsize=7, fontweight="bold")
        for m in self._target_markers:
            x, y, z = m["pos"]
            self.ax.scatter([x], [y], [z], c="#FF8800", s=280, zorder=8,
                            depthshade=False, marker="o", alpha=0.25)
            self.ax.scatter([x], [y], [z], c="#FF8800", s=70, zorder=9,
                            depthshade=False, marker="+")
            self.ax.scatter([x], [y], [z], c="#FF8800", s=30, zorder=9,
                            depthshade=False, marker="o")
            self.ax.text(x + 14, y + 14, z + 14,
                         f"[TGT] {m['name']}", color="#FF8800", fontsize=7, fontweight="bold")

    # ── Cleanup ────────────────────────────────────────────────────────

    def destroy(self):
        plt.close(self.fig)
