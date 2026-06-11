"""
3D Viewport for the robot simulation using matplotlib.

Renders the FANUC LR Mate 200iD/14L with realistic geometry,
route waypoints, user frame axes, TCP marker, and workspace boundary.
"""
from __future__ import annotations

import os
from typing import Optional, List, TYPE_CHECKING

import numpy as np

def _load_stl_file(path: str) -> Optional[np.ndarray]:
    """Load STL (binary or ASCII) without external libraries.
    Returns (N,3,3) array of triangle vertices, or None on failure."""
    import struct
    try:
        with open(path, "rb") as f:
            header = f.read(80)
            if len(header) < 80:
                return None
            # Try binary STL
            data = f.read(4)
            if len(data) < 4:
                return None
            n_tri = struct.unpack("<I", data)[0]
            expected = n_tri * 50
            raw = f.read(expected)
            if len(raw) == expected:
                tris = []
                offset = 0
                for _ in range(n_tri):
                    offset += 12  # skip normal
                    v1 = struct.unpack_from("<3f", raw, offset); offset += 12
                    v2 = struct.unpack_from("<3f", raw, offset); offset += 12
                    v3 = struct.unpack_from("<3f", raw, offset); offset += 12
                    offset += 2   # skip attribute
                    tris.append([v1, v2, v3])
                return np.array(tris, dtype=np.float32)
        # Fallback: ASCII STL
        verts = []
        with open(path, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("vertex"):
                    parts = line.split()
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if len(verts) >= 3 and len(verts) % 3 == 0:
            return np.array(verts, dtype=np.float32).reshape(-1, 3, 3)
    except Exception:
        pass
    return None
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
        self._pan_cx: float = 0.0
        self._pan_cy: float = 0.0
        self._pan_cz: float = 300.0   # 視点注視点Z（地面0固定だとズーム時に上下ドリフト）
        self._rotate_start = None  # (x, y, elev0, azim0)
        self._pan_start    = None  # (x, y, cx0, cy0, cz0)

        self._stl_verts: Optional[np.ndarray] = None   # (N,3,3) STL triangles
        self._stl_name: str = ""
        self._stl_T: np.ndarray = np.eye(4)
        self._csv_points: Optional[np.ndarray] = None  # (N,3) CSV points
        self._csv_name: str = ""
        self._csv_T: np.ndarray = np.eye(4)

        self._tcp_markers: List[dict] = []    # [{"name": str, "pos": np.ndarray}]
        self._target_markers: List[dict] = [] # [{"name": str, "pos": np.ndarray}]
        self._ref_frames: list = []  # [{"name": str, "T": np.ndarray, "color": str}]

        # 刃先CSV（ツールローカル座標・フランジ追従）
        self._blade_pts: Optional[np.ndarray] = None      # (N,3) local points
        self._blade_normals: Optional[np.ndarray] = None  # (N,3) local normals
        self._blade_name: str = ""
        self._blade_T: np.ndarray = np.eye(4)             # local offset from flange

        self.fig = plt.figure(facecolor="#161B22")
        self.fig.subplots_adjust(left=-0.18, right=1.18, bottom=-0.08, top=1.08)
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

        self.canvas.mpl_connect("scroll_event",         self._on_scroll)
        self.canvas.mpl_connect("button_press_event",   self._on_mpress)
        self.canvas.mpl_connect("button_release_event", self._on_mrelease)
        self.canvas.mpl_connect("motion_notify_event",  self._on_mmove)
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

    # ── カメラベクトル（RoboDK風の直感的操作の基盤） ──────────────────────
    def _cam_vectors(self):
        """方位角・仰角から、画面右方向・画面上方向のワールドベクトルを返す。"""
        a = np.deg2rad(self._azim)
        e = np.deg2rad(self._elev)
        ca, sa = np.cos(a), np.sin(a)
        ce, se = np.cos(e), np.sin(e)
        right = np.array([-sa,      ca,       0.0])
        up    = np.array([-se * ca, -se * sa, ce])
        return right, up

    def _px_per_world(self) -> float:
        """ワールド1mmあたりの画面ピクセル数（おおよそ）。"""
        lim = 700.0 * self._zoom_scale
        fig_w = self.fig.get_figwidth() * self.fig.dpi
        return max(fig_w, 1.0) / (2.0 * lim)

    def _center_disp(self):
        """注視点 (pan_cx, pan_cy, pan_cz) の画面ピクセル座標。"""
        from mpl_toolkits.mplot3d import proj3d
        xs, ys, _ = proj3d.proj_transform(
            self._pan_cx, self._pan_cy, self._pan_cz, self.ax.get_proj())
        return self.ax.transData.transform((xs, ys))

    def _world_under_cursor(self, px, py):
        """カーソル位置を、注視点を通る画面平行面上のワールド点に逆投影する。"""
        if px is None or py is None:
            return None
        try:
            cpx, cpy = self._center_disp()
        except Exception:
            return None
        right, up = self._cam_vectors()
        s = 1.0 / self._px_per_world()   # world per pixel
        c = np.array([self._pan_cx, self._pan_cy, self._pan_cz], float)
        return c + (px - cpx) * s * right + (py - cpy) * s * up

    def _on_scroll(self, event):
        old = self._zoom_scale
        factor = 0.85 if event.button == "up" else 1.18
        new = float(np.clip(old * factor, 0.05, 5.0))
        if new == old:
            return
        r = new / old
        # カーソル下のワールド点を固定したままズーム（RoboDK風）
        W = self._world_under_cursor(event.x, event.y)
        if W is not None:
            self._pan_cx = float(np.clip(W[0] + r * (self._pan_cx - W[0]), -3000, 3000))
            self._pan_cy = float(np.clip(W[1] + r * (self._pan_cy - W[1]), -3000, 3000))
            self._pan_cz = float(np.clip(W[2] + r * (self._pan_cz - W[2]), -2000, 4000))
        self._zoom_scale = new
        self._redraw()

    def _on_mpress(self, event):
        if event.button == 1:
            self._rotate_start = (event.x, event.y, self._elev, self._azim)
        elif event.button in (2, 3):   # 右ボタン or ホイール（中）ボタン = パン
            self._pan_start = (event.x, event.y,
                               self._pan_cx, self._pan_cy, self._pan_cz)

    def _on_mrelease(self, event):
        self._rotate_start = None
        self._pan_start    = None

    def _on_mmove(self, event):
        if self._rotate_start is not None and event.button == 1:
            dx = event.x - self._rotate_start[0]
            dy = event.y - self._rotate_start[1]
            self._azim = self._rotate_start[3] - dx * 0.5
            self._elev = float(np.clip(self._rotate_start[2] + dy * 0.5, -89.0, 89.0))
            self._redraw()
        elif self._pan_start is not None and event.button in (2, 3):
            # 掴んだ点がカーソルに追従する画面平面パン（上下ドラッグでZも移動）
            right, up = self._cam_vectors()
            s = 1.0 / self._px_per_world()
            dpx = (event.x - self._pan_start[0]) * s
            dpy = (event.y - self._pan_start[1]) * s
            delta = dpx * right + dpy * up
            self._pan_cx = float(np.clip(self._pan_start[2] - delta[0], -3000, 3000))
            self._pan_cy = float(np.clip(self._pan_start[3] - delta[1], -3000, 3000))
            self._pan_cz = float(np.clip(self._pan_start[4] - delta[2], -2000, 4000))
            self._redraw()

    def _redraw(self):
        self.ax.cla()
        self._setup_axes()
        self.ax.view_init(elev=self._elev, azim=self._azim)
        self._draw_workspace()
        self._draw_user_frame()
        self._draw_robot(self._joint_angles)
        self._draw_overlay()
        self._draw_ref_frames()
        self._draw_markers()
        self._draw_route()
        self._draw_jog_target()
        self.canvas.draw_idle()

    def _setup_axes(self):
        ax = self.ax
        ax.set_facecolor("#0D1117")

        lim = 700 * self._zoom_scale
        zhalf = lim   # 立方体ボックス（各軸スケール均一→カーソル追従が正確）
        ax.set_xlim(self._pan_cx - lim, self._pan_cx + lim)
        ax.set_ylim(self._pan_cy - lim, self._pan_cy + lim)
        ax.set_zlim(self._pan_cz - zhalf, self._pan_cz + zhalf)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

        ax.set_xlabel("X [mm]", color="#8B949E", fontsize=7, labelpad=2)
        ax.set_ylabel("Y [mm]", color="#8B949E", fontsize=7, labelpad=2)
        ax.set_zlabel("Z [mm]", color="#8B949E", fontsize=7, labelpad=2)

        step = int(lim / 3 / 100) * 100 or 100
        ticks = list(range(int(self._pan_cx - lim), int(self._pan_cx + lim) + 1, step))
        yticks = list(range(int(self._pan_cy - lim), int(self._pan_cy + lim) + 1, step))
        z0 = int((self._pan_cz - zhalf) // step) * step
        zticks = list(range(z0, int(self._pan_cz + zhalf) + 1, step))
        ax.set_xticks(ticks); ax.set_yticks(yticks); ax.set_zticks(zticks)
        ax.tick_params(colors="#555E6A", labelsize=6, length=2, pad=1)
        ax.xaxis.set_tick_params(labelcolor="#555E6A")
        ax.yaxis.set_tick_params(labelcolor="#555E6A")
        ax.zaxis.set_tick_params(labelcolor="#555E6A")

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
        self._draw_blade_csv(T_ee)
        self._draw_tcp(q, T_ee)

    def _blade_axes(self, T_ee: np.ndarray):
        """刃先CSV取付フレームの原点・刃長軸・刃幅軸・刃長を返す。

        刃先CSVは局所 Y 軸方向に刃渡りが伸び（0〜約170mm）、
        局所 Z 軸方向に刃幅をもつ。包丁モデル・TCP はこの軸に整列させる。
        """
        T = T_ee @ self._blade_T
        origin = T[:3, 3]
        R = T[:3, :3]
        blade_dir  = R[:, 1]   # 刃渡り方向（局所 +Y）
        width_dir  = R[:, 2]   # 刃幅方向（局所 +Z）
        if self._blade_pts is not None and len(self._blade_pts):
            blade_len = float(np.max(self._blade_pts[:, 1]))
            if blade_len < 1.0:
                blade_len = KNIFE_BLADE_LEN
        else:
            blade_len = KNIFE_BLADE_LEN
        return origin, blade_dir, width_dir, blade_len

    def _draw_knife(self, q: np.ndarray, T_ee: np.ndarray):
        """Draw simplified knife model aligned with the blade-CSV axis.

        包丁は刃先CSVの取付オフセット（_blade_T）に追従し、刃渡り方向（局所Y）に
        整列させて描画する。柄はフランジから刃元へ橋渡しする。
        """
        flange = T_ee[:3, 3]
        origin, blade_dir, width_dir, blade_len = self._blade_axes(T_ee)

        # 柄: フランジ → 刃元（origin）
        self.ax.plot([flange[0], origin[0]],
                     [flange[1], origin[1]],
                     [flange[2], origin[2]],
                     color=KNIFE_HANDLE, lw=5, solid_capstyle="round")

        # 刃: 刃元 → 刃先（刃渡り方向）
        blade_tip = origin + blade_len * blade_dir
        self.ax.plot([origin[0], blade_tip[0]],
                     [origin[1], blade_tip[1]],
                     [origin[2], blade_tip[2]],
                     color=KNIFE_BLADE, lw=2.5, solid_capstyle="round")

        hw = KNIFE_BLADE_WIDTH / 2
        corners = np.array([
            origin    - hw * width_dir, origin    + hw * width_dir,
            blade_tip + hw * width_dir, blade_tip - hw * width_dir,
        ])
        poly = Poly3DCollection([corners], alpha=0.22,
                                facecolor=KNIFE_BLADE,
                                edgecolor="#666666", linewidth=0.5)
        self.ax.add_collection3d(poly)

    def _draw_tcp(self, q: np.ndarray, T_ee: np.ndarray):
        """Draw TCP marker. 刃先CSVがあれば刃先端へ、無ければツールフレームへ。"""
        flange = T_ee[:3, 3]
        if self._blade_pts is not None and len(self._blade_pts):
            origin, blade_dir, _w, blade_len = self._blade_axes(T_ee)
            tcp_pos = origin + blade_len * blade_dir
        elif self._tool_frame is not None and self._tool_frame.z != 0.0:
            T_tcp   = T_ee @ self._tool_frame.to_transform()
            tcp_pos = T_tcp[:3, 3]
        else:
            return

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
        verts = _load_stl_file(path)
        if verts is None:
            return False
        self._stl_verts = verts
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

    # ── 刃先CSV（フランジ追従） ────────────────────────────────────────

    def load_blade_csv(self, path: str) -> int:
        """刃先CSV（x,y,z,nx,ny,nz 6列・ヘッダーなし）を読み込み、
        ナイフ先端に追従するローカル点群として保持する。

        Returns: 読み込んだ点数（0=失敗）。
        """
        import csv
        pts, nrm = [], []
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                for row in csv.reader(f):
                    if len(row) >= 6:
                        try:
                            vals = [float(v) for v in row[:6]]
                        except ValueError:
                            continue
                        pts.append(vals[:3])
                        nrm.append(vals[3:6])
        except OSError:
            return 0
        if not pts:
            return 0
        self._blade_pts     = np.array(pts, dtype=float)
        self._blade_normals = np.array(nrm, dtype=float)
        self._blade_name    = os.path.basename(path)
        self._redraw()
        return len(pts)

    def set_blade_pose(self, x, y, z, rx, ry, rz):
        """フランジから刃先CSVローカル原点へのオフセットを設定する。"""
        from ..robot.kinematics import Kinematics
        self._blade_T = Kinematics.pose_to_transform(x, y, z, rx, ry, rz)
        self._redraw()

    def clear_blade(self):
        self._blade_pts = None
        self._blade_normals = None
        self._blade_name = ""
        self._blade_T = np.eye(4)
        self._redraw()

    def has_blade(self) -> bool:
        return self._blade_pts is not None

    def _draw_blade_csv(self, T_ee: np.ndarray):
        """刃先CSV点群をフランジ姿勢に追従させて描画する。"""
        if self._blade_pts is None:
            return
        T = T_ee @ self._blade_T
        R, t = T[:3, :3], T[:3, 3]
        pts = (R @ self._blade_pts.T).T + t
        self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                        c="#FF5577", s=4, alpha=0.85, depthshade=False)
        # 法線ウィスカー（間引き表示・研磨接触方向の確認用）
        if self._blade_normals is not None:
            nrm = (R @ self._blade_normals.T).T
            for p, n in zip(pts[::8], nrm[::8]):
                tip = p + 8.0 * n
                self.ax.plot([p[0], tip[0]], [p[1], tip[1]], [p[2], tip[2]],
                             color="#FF99AA", lw=0.5, alpha=0.5)
        ctr = pts.mean(axis=0)
        self.ax.text(ctr[0] + 10, ctr[1] + 10, ctr[2] + 10,
                     f"{self._blade_name} ({len(pts)} pts)",
                     color="#FF7799", fontsize=6, alpha=0.9)

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

            # 三角形を間引いてソリッド面で描画（法線による簡易シェーディング）
            max_tris = 1500
            step = max(1, len(tverts) // max_tris)
            tris = tverts[::step]

            v1 = tris[:, 1] - tris[:, 0]
            v2 = tris[:, 2] - tris[:, 0]
            normals = np.cross(v1, v2)
            lens = np.linalg.norm(normals, axis=1, keepdims=True)
            lens[lens < 1e-9] = 1.0
            normals /= lens

            light = np.array([0.4, -0.3, 0.85])
            light /= np.linalg.norm(light)
            intensity = 0.35 + 0.65 * np.abs(normals @ light)

            base = np.array([0.45, 0.58, 0.75])  # 青灰色（機械色）
            facecolors = np.clip(base[None, :] * intensity[:, None], 0, 1)

            poly = Poly3DCollection(tris, facecolors=facecolors,
                                    edgecolors="none", alpha=0.95)
            self.ax.add_collection3d(poly)

            ctr = tv.mean(axis=0)
            zmax = tv[:, 2].max()
            self.ax.text(ctr[0], ctr[1], zmax + 25,
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

    # ── Reference Frames ───────────────────────────────────────────────

    def add_ref_frame(self, name: str, x, y, z, rx, ry, rz, color="#FF88FF"):
        """Add a named reference frame displayed as XYZ axes in the 3D viewport."""
        from ..robot.kinematics import Kinematics
        T = Kinematics.pose_to_transform(x, y, z, rx, ry, rz)
        self._ref_frames.append({"name": name, "T": T, "color": color})
        self._redraw()

    def remove_ref_frame(self, name: str):
        """Remove a reference frame by name."""
        self._ref_frames = [f for f in self._ref_frames if f["name"] != name]
        self._redraw()

    def clear_ref_frames(self):
        """Remove all reference frames."""
        self._ref_frames.clear()
        self._redraw()

    def get_ref_frames(self) -> list:
        """Return a copy of the current reference frame list."""
        return list(self._ref_frames)

    def _draw_ref_frames(self):
        """Draw all named reference frames as XYZ axis triads with labels."""
        scale = 80
        for rf in self._ref_frames:
            T = rf["T"]
            origin = T[:3, 3]
            R = T[:3, :3]
            base_color = rf.get("color", "#FF88FF")
            for col, clr in enumerate(["#FF4444", "#44FF44", "#4444FF"]):
                tip = origin + scale * R[:, col]
                self.ax.plot([origin[0], tip[0]], [origin[1], tip[1]],
                             [origin[2], tip[2]], color=clr, lw=2.5, alpha=0.9)
            self.ax.scatter([origin[0]], [origin[1]], [origin[2]],
                            c=base_color, s=80, zorder=8, depthshade=False, marker="D")
            self.ax.text(origin[0]+12, origin[1]+12, origin[2]+12,
                         rf["name"], color=base_color, fontsize=7, fontweight="bold")

    # ── Cleanup ────────────────────────────────────────────────────────

    def destroy(self):
        plt.close(self.fig)
