"""
3D Viewport for the robot simulation using matplotlib.

Renders the FANUC LR Mate 200iD/14L with realistic geometry,
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
        """
        Draw FANUC LR Mate 200iD/14L with realistic geometry.

        Key visual features of the actual robot:
          - Wide lateral shoulder housing (most distinctive feature)
          - Similar elbow housing at J3
          - Yellow throughout; dark accents only at joint interfaces
          - Slender cylindrical forearm and wrist
        """
        pos = self.kin.get_joint_positions(q)  # (7,3) base + J1…J6
        tfs = self.kin.forward_all(q)           # list of 4x4 transforms

        p0 = pos[0]   # base origin
        p1 = pos[1]   # J2 shoulder axis
        p2 = pos[2]   # J3 elbow axis
        p3 = pos[3]   # J4 wrist-roll axis
        p4 = pos[4]   # J5 wrist-pitch axis
        p5 = pos[5]   # J6 wrist-spin axis
        p6 = pos[6]   # flange face

        # Joint frame rotations — used to orient housings correctly
        R1 = tfs[1][:3, :3]   # J1 rotating frame
        R2 = tfs[2][:3, :3]   # J2/upper-arm frame (for elbow housing)

        ax = self.ax

        # ─── ① ベース台座（固定・黄色ボックス） ─────────────────────────
        # 実機: 幅広い矩形台座。全体が黄色。
        _rotated_box(ax, p0 + [0, 0, 44], np.eye(3),
                     lx=200, ly=160, lz=88, color=FANUC_YELLOW)
        # 底面リップ（ダーク）
        _rotated_box(ax, p0 + [0, 0, 5], np.eye(3),
                     lx=232, ly=192, lz=10, color=FANUC_DARK_GRAY)

        # ─── ② J1 回転胴体（ベース上〜肩下まで） ────────────────────────
        # 実機: 台座から立ち上がる黄色の太いドラム形ボディ。J1と共に回転。
        # 中心: J1フレーム内でZ方向185mm上 → [0,0,185] in J1 frame
        body_ctr = p0 + R1 @ np.array([0.0, 0.0, 185.0])
        _rotated_box(ax, body_ctr, R1, lx=120, ly=105, lz=190, color=FANUC_YELLOW)

        # ─── ③ 肩ハウジング（最も特徴的・横長ボックス） ────────────────
        # 実機: J2軸を中心に左右230mm張り出す幅広ハウジング。
        # FANUCロボットの最も目立つ外観特徴。
        _rotated_box(ax, p1, R1, lx=80, ly=230, lz=105, color=FANUC_YELLOW)
        # 両側の丸みキャップ（側面ディスク）
        _disk(ax, p1 + 115 * R1[:, 1],  R1[:, 1], 53, FANUC_YELLOW_D)
        _disk(ax, p1 - 115 * R1[:, 1], -R1[:, 1], 53, FANUC_YELLOW_D)

        # ─── ④ 上腕リンク（J2→J3・矩形断面） ───────────────────────────
        # 実機: 肩から肘まで長い矩形断面アーム（ボックス形状）
        _box_link(ax, p1, p2, w=80, h=65, color=FANUC_YELLOW)

        # ─── ⑤ 肘ハウジング（J3・肩と同形の横長ボックス） ───────────────
        # 実機: 肘部にも肩と同様の横長ハウジングがある（やや小型）
        _rotated_box(ax, p2, R2, lx=60, ly=160, lz=72, color=FANUC_YELLOW)
        _disk(ax, p2 + 80 * R2[:, 1],  R2[:, 1], 44, FANUC_YELLOW_D)
        _disk(ax, p2 - 80 * R2[:, 1], -R2[:, 1], 44, FANUC_YELLOW_D)

        # ─── ⑥ 前腕（J3→J4・円柱形） ───────────────────────────────────
        # 実機: 肘から手首に向かって細くなる円筒形
        _cylinder(ax, p2, p3, 34, FANUC_YELLOW, n=14)
        _disk(ax, p2, -(p3 - p2), 34, FANUC_YELLOW_D)
        _disk(ax, p3,   p3 - p2,  34, FANUC_YELLOW_D)

        # ─── ⑦ 手首 J4（球体） + J4→J5 ─────────────────────────────────
        _sphere(ax, p3, 34, FANUC_BLACK, n=10)
        _cylinder(ax, p3, p4, 26, FANUC_YELLOW, n=12)
        _disk(ax, p4, p4 - p3, 26, FANUC_YELLOW_D)

        # ─── ⑧ 手首 J5（球体） + J5→J6 ─────────────────────────────────
        _sphere(ax, p4, 26, FANUC_BLACK, n=10)
        _cylinder(ax, p4, p5, 21, FANUC_YELLOW, n=12)
        _disk(ax, p5, p5 - p4, 21, FANUC_YELLOW_D)

        # ─── ⑨ 手首 J6（球体） + フランジ ───────────────────────────────
        _sphere(ax, p5, 21, FANUC_BLACK, n=8)
        _cylinder(ax, p5, p6, 18, FANUC_YELLOW, n=12)
        ee_dir = p6 - p5
        if np.linalg.norm(ee_dir) > 1e-3:
            nd = ee_dir / np.linalg.norm(ee_dir)
            _disk(ax, p6,  nd, 28, FANUC_BLACK, alpha=0.95)
            _cylinder(ax, p6, p6 + nd * 12, 28, FANUC_DARK_GRAY, n=12)

        # ─── 地面への影 ──────────────────────────────────────────────────
        ax.plot(pos[:, 0], pos[:, 1], np.zeros(len(pos)),
                color="#444444", lw=2.5, alpha=0.2)

        # ─── EE 座標フレーム ──────────────────────────────────────────────
        T_ee = self.kin.forward(q)
        origin = T_ee[:3, 3]
        R = T_ee[:3, :3]
        scale = 65
        for col, (color, name) in enumerate(
                zip(["#FF4444", "#44FF44", "#4444FF"], ["X", "Y", "Z"])):
            tip = origin + scale * R[:, col]
            ax.plot([origin[0], tip[0]], [origin[1], tip[1]],
                    [origin[2], tip[2]], color=color, lw=2.0, alpha=0.9)
            ax.text(tip[0], tip[1], tip[2], name,
                    color=color, fontsize=6, alpha=0.85)

        # ─── 包丁・TCP ───────────────────────────────────────────────────
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

    # ── Cleanup ────────────────────────────────────────────────────────

    def destroy(self):
        plt.close(self.fig)
