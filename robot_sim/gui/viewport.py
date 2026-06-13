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
                # 50バイト/三角形（法線12 + 頂点36 + 属性2）を一括デコード
                rec = np.dtype([("normal", "<f4", (3,)),
                                ("verts",  "<f4", (3, 3)),
                                ("attr",   "<u2")])
                tris = np.frombuffer(raw, dtype=rec, count=n_tri)["verts"]
                return np.ascontiguousarray(tris, dtype=np.float32)
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

# ── ロボット実機メッシュ（ROS-Industrial LR Mate 200iD/7L 形状・mm 単位） ──
# 各リンクの (メッシュファイル名, ベースカラー RGB)。
# link_2 / link_4 はロングアーム（7L/14L 共通外形）専用メッシュ。
_ROBOT_MESH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "assets", "robot")
_ROBOT_LINKS = [
    ("base_link", (0.282, 0.301, 0.317)),   # ベース: グレー
    ("link_1",    (0.960, 0.768, 0.000)),   # J1〜J5: FANUC イエロー
    ("link_2",    (0.960, 0.768, 0.000)),
    ("link_3",    (0.960, 0.768, 0.000)),
    ("link_4",    (0.960, 0.768, 0.000)),
    ("link_5",    (0.960, 0.768, 0.000)),
    ("link_6",    (0.180, 0.180, 0.190)),   # フランジ: ブラック
]


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


class Viewport3D:
    """Embedded 3D matplotlib viewport inside a tkinter frame."""

    # 立方体データボックスを subplot 矩形より一回り小さく描画する係数。
    # 1.0 だと回転・パン時にボックス隅が矩形外へはみ出して STL 等が
    # 見切れる。< 1.0 で余白を確保し端の見切れを防ぐ。
    _BOX_ZOOM = 0.78

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
        self._stl_path: str = ""
        self._stl_T: np.ndarray = np.eye(4)
        self._csv_points: Optional[np.ndarray] = None  # (N,3) CSV points
        self._csv_name: str = ""
        self._csv_path: str = ""
        self._csv_T: np.ndarray = np.eye(4)

        self._tcp_markers: List[dict] = []    # [{"name": str, "pos": np.ndarray}]
        self._target_markers: List[dict] = [] # [{"name": str, "pos": np.ndarray}]
        self._ref_frames: list = []  # [{"name": str, "T": np.ndarray, "color": str}]

        # 刃先CSV（ツールローカル座標・フランジ追従）
        self._blade_pts: Optional[np.ndarray] = None      # (N,3) local points
        self._blade_normals: Optional[np.ndarray] = None  # (N,3) local normals
        self._blade_name: str = ""
        self._blade_path: str = ""
        self._blade_T: np.ndarray = np.eye(4)             # local offset from flange

        # 選択可能曲線（RoboDK風 曲線選択ダイアログ用）
        self._pick_curves: List[np.ndarray] = []      # [(M,3) pts]
        self._pick_curves_local = False               # True=刃先ローカル(包丁追従)
        self._pick_orders: List[Optional[int]] = []   # 選択順 (1始まり) / None=未選択
        self._pick_callback = None                    # callback(curve_idx)
        self._pick_artist_map: dict = {}              # id(artist) -> curve_idx
        self._pick_candidate: Optional[int] = None    # press時のピック候補

        # 実機メッシュ（assets/robot/*.stl）— 読込失敗時は円柱フォールバック
        self._link_meshes: list = []   # [(verts (N,3,3), normals (N,3), rgb)]
        self._fast_mode: bool = False  # 再生中は軽量表示（円柱）へ切替
        self._pre_img = None           # 事前描画再生中の figimage（None=通常描画）
        self._load_robot_meshes()

        self.fig = plt.figure(facecolor="#161B22")
        self.fig.subplots_adjust(left=-0.18, right=1.18, bottom=-0.08, top=1.08)
        self.ax: Axes3D = self.fig.add_subplot(111, projection="3d")
        self._setup_axes()

        # 向きインジケータ（XYZ軸ギズモ）— 画面左下隅に固定表示。
        # 本体ビューの回転（elev/azim）に追従し、回転しても常に X/Y/Z が見える。
        self._gizmo_ax = self.fig.add_axes(
            (0.012, 0.012, 0.15, 0.15), projection="3d")
        self._gizmo_ax.set_navigate(False)
        self._gizmo_ax.patch.set_alpha(0.0)

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
        self.canvas.mpl_connect("pick_event",           self._on_pick)
        self.update_robot(self._joint_angles)

    # ── Public interface ───────────────────────────────────────────────

    def update_robot(self, joint_angles: np.ndarray):
        self._joint_angles = np.asarray(joint_angles)
        self._redraw()

    def set_fast_mode(self, enabled: bool):
        """軽量表示（円柱ジオメトリ）の ON/OFF を切り替える。

        True: 実機メッシュを使わず円柱ジオメトリで描画し描画時間を短縮。
        False: 実機メッシュで描画。
        再生中・停止中いずれでも呼べ、状態が変われば即座に再描画する。
        """
        if self._fast_mode == enabled:
            return
        self._fast_mode = enabled
        self._redraw()  # 表示モードを即時反映

    def set_route(self, route: Optional["Route"]):
        self._route = route
        self._redraw()

    def set_selected_waypoint(self, idx: Optional[int]):
        if idx == self._selected_wp_idx:
            return  # 同一選択は再描画しない（シミュ再生中の冗長再描画防止）
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
        # set_box_aspect(zoom=_BOX_ZOOM) で内容が縮小される分を反映する
        # （カーソル追従ズームのスケール整合のため）。
        return self._BOX_ZOOM * max(fig_w, 1.0) / (2.0 * lim)

    def _center_disp(self):
        """3D ビューの画面中心ピクセル座標（axes bbox の中心を使用）。

        matplotlib 3D の proj3d + transData 経由の逆投影は環境依存で失敗
        することがあるため、axes の position bbox から直接計算する。
        pan_cx/cy/cz の注視点は常に画面中央に描画されるため、这の近似は正確。
        """
        bbox = self.ax.get_position()
        fw = self.fig.get_figwidth() * self.fig.dpi
        fh = self.fig.get_figheight() * self.fig.dpi
        return (bbox.x0 + bbox.x1) * 0.5 * fw, (bbox.y0 + bbox.y1) * 0.5 * fh

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
        # クリック（ドラッグ距離 < 5px）なら曲線ピックとして処理。
        # ドラッグ（>= 5px）は従来どおり視点回転のみ（ピック候補は破棄）。
        if (event.button == 1 and self._pick_candidate is not None
                and self._rotate_start is not None
                and self._pick_callback is not None):
            dx = event.x - self._rotate_start[0]
            dy = event.y - self._rotate_start[1]
            if dx * dx + dy * dy < 25.0:   # 5px 未満 → クリック扱い
                idx = self._pick_candidate
                cb  = self._pick_callback
                self._pick_candidate = None
                self._rotate_start = None
                self._pan_start    = None
                cb(idx)
                return
        self._pick_candidate = None
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

    def _draw_scene(self):
        """現在の状態で 3D シーンを完全に再構築する（canvas へは描画しない）。"""
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
        self._draw_pick_curves()
        self._draw_jog_target()
        self._draw_gizmo()

    def _draw_gizmo(self):
        """左下隅の向きインジケータ（X=赤 / Y=緑 / Z=青）を本体ビューに同期描画。

        本体ビューの elev/azim に追従するだけで、パン・ズームには影響されない。
        回転しても常に画面隅に XYZ 軸が表示され続ける。
        """
        g = self._gizmo_ax
        g.cla()
        g.set_navigate(False)
        g.patch.set_alpha(0.0)
        L = 1.0
        # 原点からの3軸（R/G/B）
        axes = (((L, 0, 0), "#FF4D4D", "X"),
                ((0, L, 0), "#4DFF77", "Y"),
                ((0, 0, L), "#5599FF", "Z"))
        for (vx, vy, vz), col, name in axes:
            g.plot([0, vx], [0, vy], [0, vz], color=col, lw=2.2)
            g.text(vx * 1.35, vy * 1.35, vz * 1.35, name,
                   color=col, fontsize=8, fontweight="bold",
                   ha="center", va="center")
        g.set_xlim(-L, L); g.set_ylim(-L, L); g.set_zlim(-L, L)
        try:
            g.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        g.view_init(elev=self._elev, azim=self._azim)
        # 軸装飾をすべて消して矢印だけ見せる
        g.set_axis_off()

    def _redraw(self):
        # 事前描画再生中は 3D シーンを触らない（figimage を表示し続ける）
        if self._pre_img is not None:
            return
        self._draw_scene()
        self.canvas.draw_idle()

    # ── 事前描画（プリレンダリング）再生 ──────────────────────────────
    # 案1: 再生前に全フレームをオフスクリーン描画して RGBA 画像として保持し、
    # 再生中は重い 3D 再描画をやめて画像を差し替えるだけにする。

    def render_frame(self, joint_angles) -> np.ndarray:
        """与えた関節角でシーンを完全描画し RGBA 画像 (H,W,4 uint8) を返す。

        通常品質（実機メッシュ等、現在の表示設定）でレンダリングする。
        コストはここで一度だけ払い、再生は画像差し替えのみで滑らかになる。
        """
        self._joint_angles = np.asarray(joint_angles)
        self._draw_scene()
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba()).copy()

    def begin_prerendered_playback(self, first_frame: np.ndarray):
        """事前描画再生を開始: 3D 軸を隠し figimage を 1 枚用意する。"""
        if self._pre_img is not None:
            self.end_prerendered_playback()
        self.ax.set_visible(False)
        self._gizmo_ax.set_visible(False)  # ギズモはフレーム画像側に焼き込み済み
        self._pre_img = self.fig.figimage(
            first_frame, xo=0, yo=0, origin="upper", zorder=10)
        self.canvas.draw()

    def show_prerendered_frame(self, frame: np.ndarray):
        """事前描画フレームを表示（3D 再描画なし・画像差し替えのみ）。"""
        if self._pre_img is None:
            return
        self._pre_img.set_data(frame)
        self.canvas.draw()

    def end_prerendered_playback(self):
        """事前描画再生を終了し通常の 3D 描画へ戻す。"""
        if self._pre_img is None:
            return
        try:
            self._pre_img.remove()
        except Exception:
            pass
        self._pre_img = None
        self.ax.set_visible(True)
        self._gizmo_ax.set_visible(True)
        self._redraw()

    def _setup_axes(self):
        ax = self.ax
        ax.set_facecolor("#0D1117")

        lim = 700 * self._zoom_scale
        zhalf = lim   # 立方体ボックス（各軸スケール均一→カーソル追従が正確）
        ax.set_xlim(self._pan_cx - lim, self._pan_cx + lim)
        ax.set_ylim(self._pan_cy - lim, self._pan_cy + lim)
        ax.set_zlim(self._pan_cz - zhalf, self._pan_cz + zhalf)
        try:
            # zoom<1 で 3D 内容を subplot 矩形内へ収め、端の見切れを防ぐ。
            ax.set_box_aspect((1, 1, 1), zoom=self._BOX_ZOOM)
        except TypeError:
            # 旧 matplotlib（zoom 非対応）フォールバック
            try:
                ax.set_box_aspect((1, 1, 1))
            except Exception:
                pass
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

    # ── 実機メッシュ描画（ROS-Industrial LR Mate 200iD/7L） ────────────

    def _load_robot_meshes(self):
        """assets/robot/ の実機リンクメッシュを読み込み、法線を事前計算する。"""
        meshes = []
        for name, rgb in _ROBOT_LINKS:
            path = os.path.join(_ROBOT_MESH_DIR, name + ".stl")
            verts = _load_stl_file(path)
            if verts is None:
                self._link_meshes = []   # 1つでも欠けたら全てフォールバック
                return
            v1 = verts[:, 1] - verts[:, 0]
            v2 = verts[:, 2] - verts[:, 0]
            normals = np.cross(v1, v2)
            lens = np.linalg.norm(normals, axis=1, keepdims=True)
            lens[lens < 1e-12] = 1.0
            normals = (normals / lens).astype(np.float32)
            meshes.append((verts, normals, np.asarray(rgb, dtype=float)))
        self._link_meshes = meshes

    def _urdf_link_transforms(self, q: np.ndarray) -> list:
        """各リンク（base_link, link_1..link_6）のワールド 4x4 変換を返す。

        メッシュは ROS-Industrial URDF（lrmate200id7l）のリンク座標系で
        定義されているため、URDF の連鎖でリンク姿勢を計算する。
        関節角は本アプリの MDH 規約 q から
        θ = (q1, q2+90°, -q3, -q4, -q5, -q6) で URDF 規約へ変換する
        （この対応で URDF tool0 と MDH フランジが全姿勢で一致する）。
        """
        t1, t2, t3, t4, t5, t6 = (q[0], q[1] + np.pi / 2,
                                  -q[2], -q[3], -q[4], -q[5])

        def rot(axis, t):
            c, s = np.cos(t), np.sin(t)
            T = np.eye(4)
            if axis == "z":
                T[:2, :2] = [[c, -s], [s, c]]
            elif axis == "y":
                T[0, 0], T[0, 2], T[2, 0], T[2, 2] = c, s, -s, c
            else:  # x
                T[1, 1], T[1, 2], T[2, 1], T[2, 2] = c, -s, s, c
            return T

        def tr(x, y, z):
            T = np.eye(4)
            T[:3, 3] = [x, y, z]
            return T

        Ts = [np.eye(4)]                                  # base_link
        T = tr(0, 0, 330) @ rot("z", t1);  Ts.append(T)   # link_1
        T = T @ tr(50, 0, 0) @ rot("y", t2);  Ts.append(T)   # link_2
        T = T @ tr(0, 0, 440) @ rot("y", -t3); Ts.append(T)  # link_3
        T = T @ tr(0, 0, 35) @ rot("x", -t4);  Ts.append(T)  # link_4
        T = T @ tr(420, 0, 0) @ rot("y", -t5); Ts.append(T)  # link_5
        T = T @ tr(80, 0, 0) @ rot("x", -t6);  Ts.append(T)  # link_6
        return Ts

    _LIGHT_DIR = np.array([0.45, -0.35, 0.82]) / np.linalg.norm([0.45, -0.35, 0.82])

    def _draw_robot_meshes(self, q: np.ndarray):
        """実機メッシュをランバートシェーディング付きで描画する。

        全リンクを1つの Poly3DCollection に統合する — matplotlib の
        Zソートはコレクション内でのみ働くため、リンク間の前後関係を
        正しく描画するには統合が必須。
        """
        Ts = self._urdf_link_transforms(q)
        all_tris = []
        all_colors = []
        for (verts, normals, rgb), T in zip(self._link_meshes, Ts):
            R, t = T[:3, :3], T[:3, 3]
            all_tris.append(verts @ R.T + t)     # (N,3,3) ワールド座標へ
            tn = normals @ R.T                   # (N,3)  回転のみ
            inten = 0.30 + 0.70 * np.abs(tn @ self._LIGHT_DIR)
            all_colors.append(np.clip(rgb[None, :] * inten[:, None], 0, 1))
        poly = Poly3DCollection(np.concatenate(all_tris),
                                facecolors=np.concatenate(all_colors),
                                edgecolors="none", alpha=1.0)
        poly.set_zsort("average")
        self.ax.add_collection3d(poly)

    def _draw_robot(self, q: np.ndarray):
        """Draw FANUC LR Mate 200iD/14L（実機メッシュ、欠落時は円柱形状）。"""
        pos = self.kin.get_joint_positions(q)  # (7, 3)  Base + J1…J6

        if self._link_meshes and not self._fast_mode:
            self._draw_robot_meshes(q)

            # 地面の影（リンク原点の投影ポリライン）
            self.ax.plot(pos[:, 0], pos[:, 1], np.zeros(len(pos)),
                         color="#333333", lw=3, alpha=0.25)

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

            self._draw_knife(T_ee)
            self._draw_blade_csv(T_ee)
            self._draw_tcp(T_ee)
            return

        # ── フォールバック: 簡易円柱ジオメトリ ───────────────────────

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

        self._draw_knife(T_ee)
        self._draw_blade_csv(T_ee)
        self._draw_tcp(T_ee)

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

    def _draw_knife(self, T_ee: np.ndarray):
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

    def _draw_tcp(self, T_ee: np.ndarray):
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

    def _draw_frame_triad(self, T: np.ndarray, scale: float,
                          axis_colors, axis_labels=None,
                          lw: float = 2.0, alpha: float = 0.85):
        """座標フレームの XYZ 軸トライアドを描画する共通ヘルパー。"""
        origin = T[:3, 3]
        R      = T[:3, :3]
        for col in range(3):
            tip = origin + scale * R[:, col]
            self.ax.plot([origin[0], tip[0]], [origin[1], tip[1]],
                         [origin[2], tip[2]],
                         color=axis_colors[col], lw=lw, alpha=alpha)
            if axis_labels:
                self.ax.text(tip[0], tip[1], tip[2],
                             axis_labels[col], color=axis_colors[col], fontsize=7)
        return origin

    def _draw_user_frame(self):
        """Draw user frame coordinate axes."""
        if self._user_frame is None:
            return
        origin = self._draw_frame_triad(
            self._user_frame.to_transform(), 120,
            ["#FF4444", "#44FF44", "#4444FF"], ["X", "Y", "Z"])

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

    # 大規模ルートしきい値: これを超えると per-point マーカー/ラベルを省略
    ROUTE_BIG_N = 25

    def _draw_route(self):
        """Draw waypoints and route path.

        大規模ルート（> ROUTE_BIG_N 点）は1本のポリライン + 始点/終点
        マーカーのみで描画する（per-point の scatter / text を作らない）。
        300点超のルートでも再描画・シミュ再生が軽量に保たれる。
        """
        if self._route is None or len(self._route) == 0:
            return

        positions = self._route.positions_array()
        n = len(positions)
        self.ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                     color=ROUTE_COLOR, lw=1.5, alpha=0.7, zorder=3)

        if n > self.ROUTE_BIG_N:
            # 軽量モード: 始点(緑)・終点(赤)と選択中の1点のみマーカー表示
            p0, p1 = positions[0], positions[-1]
            self.ax.scatter([p0[0]], [p0[1]], [p0[2]],
                            c=WP_ACTIVE, s=60, zorder=7,
                            depthshade=False, marker="o")
            self.ax.scatter([p1[0]], [p1[1]], [p1[2]],
                            c=WP_COLOR, s=60, zorder=7,
                            depthshade=False, marker="s")
            self.ax.text(p0[0] + 10, p0[1] + 10, p0[2] + 10,
                         f"START ({n}点)", color=WP_ACTIVE,
                         fontsize=6, alpha=0.85)
            self.ax.text(p1[0] + 10, p1[1] + 10, p1[2] + 10,
                         "END", color=WP_COLOR, fontsize=6, alpha=0.85)
            sel = self._selected_wp_idx
            if sel is not None and 0 <= sel < n:
                wp = self._route.waypoints[sel]
                self.ax.scatter([wp.x], [wp.y], [wp.z],
                                c=WP_ACTIVE, s=100, zorder=8,
                                depthshade=False, marker="*")
                label_text = f"{sel+1}:{wp.label}" if wp.label else f"P[{sel+1}]"
                self.ax.text(wp.x + 10, wp.y + 10, wp.z + 10,
                             label_text, color="white", fontsize=6, alpha=0.9)
            return

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
        self._stl_path = path
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
            self._csv_path = path
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
        self._blade_path    = path
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
        self._blade_path = ""
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

    def clear_stl(self):
        self._stl_verts = None
        self._stl_name = ""
        self._stl_path = ""
        self._stl_T = np.eye(4)
        self._redraw()

    def clear_csv(self):
        self._csv_points = None
        self._csv_name = ""
        self._csv_path = ""
        self._csv_T = np.eye(4)
        self._redraw()

    # ── レイヤー状態のスナップショット（Undo/Redo 用） ────────────────

    _LAYER_FIELDS = (
        "_stl_verts", "_stl_name", "_stl_path", "_stl_T",
        "_csv_points", "_csv_name", "_csv_path", "_csv_T",
        "_blade_pts", "_blade_normals", "_blade_name", "_blade_path",
        "_blade_T",
    )

    def snapshot_layers(self) -> dict:
        """STL/CSV/刃先CSV/参照フレームの状態を辞書として返す。

        点群データ（ndarray）は読み込み後に書き換えられないため参照を
        共有し、姿勢行列のみコピーする。
        """
        snap = {}
        for f in self._LAYER_FIELDS:
            v = getattr(self, f)
            snap[f] = v.copy() if f.endswith("_T") else v
        snap["ref_frames"] = [
            {"name": rf["name"], "T": rf["T"].copy(),
             "color": rf.get("color", "#FF88FF")}
            for rf in self._ref_frames
        ]
        return snap

    def restore_layers(self, snap: dict):
        """snapshot_layers() の状態を復元して再描画する。"""
        for f in self._LAYER_FIELDS:
            v = snap[f]
            setattr(self, f, v.copy() if f.endswith("_T") else v)
        self._ref_frames = [
            {"name": rf["name"], "T": rf["T"].copy(), "color": rf["color"]}
            for rf in snap["ref_frames"]
        ]
        self._redraw()

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

    # ── 選択可能曲線（RoboDK風 曲線選択） ──────────────────────────────

    def set_pick_curves(self, curves: List[np.ndarray], callback,
                        *, blade_local: bool = False):
        """クリック選択可能な曲線（ポリライン群）を設定する。

        Args:
            curves   : [(M,3) ndarray] 曲線点列リスト
            callback : callback(curve_idx) — クリック（ドラッグなし）で呼ばれる
            blade_local : True の場合、curves を刃先CSVローカル座標として扱い、
                          描画のたびに現在のフランジ姿勢 (T_ee @ _blade_T) で
                          ワールドへ変換する（包丁＝刃先オーバーレイに追従）。
                          False（既定）は従来どおりワールド座標に固定。
        """
        self._pick_curves = [np.asarray(c, dtype=float) for c in curves]
        self._pick_curves_local = blade_local
        self._pick_orders = [None] * len(self._pick_curves)
        self._pick_callback = callback
        self._redraw()

    def set_pick_orders(self, orders: List[Optional[int]]):
        """各曲線の選択順（1始まり・None=未選択）を更新する。"""
        self._pick_orders = list(orders)
        self._redraw()

    def clear_pick_curves(self):
        """選択可能曲線をすべて解除する。"""
        if not self._pick_curves and self._pick_callback is None:
            return
        self._pick_curves = []
        self._pick_curves_local = False
        self._pick_orders = []
        self._pick_callback = None
        self._pick_artist_map = {}
        self._pick_candidate = None
        self._redraw()

    def _on_pick(self, event):
        """pick_event: 押下時に候補のみ記録する（確定はリリース時の
        ドラッグ距離判定 — _on_mrelease 参照）。"""
        if getattr(event.mouseevent, "button", None) != 1:
            return
        idx = self._pick_artist_map.get(id(event.artist))
        if idx is not None and self._pick_candidate is None:
            self._pick_candidate = idx

    def _draw_pick_curves(self):
        """選択可能曲線を描画する。未選択=シアン細線 / 選択=緑太線+順番号。"""
        self._pick_artist_map = {}
        if not self._pick_curves:
            return
        # 刃先ローカル指定なら現在のフランジ姿勢でワールドへ変換（包丁追従）。
        if self._pick_curves_local:
            T = self.kin.forward(self._joint_angles) @ self._blade_T
            Rw, tw = T[:3, :3], T[:3, 3]
            curves = [(Rw @ c.T).T + tw for c in self._pick_curves]
        else:
            curves = self._pick_curves
        for i, pts in enumerate(curves):
            order = (self._pick_orders[i]
                     if i < len(self._pick_orders) else None)
            if order is not None:
                color, lw, alpha = "#00FF66", 3.0, 1.0
            else:
                color, lw, alpha = "#00CCDD", 1.2, 0.9
            ln, = self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                               color=color, lw=lw, alpha=alpha,
                               zorder=9, picker=5)
            self._pick_artist_map[id(ln)] = i
            if order is not None:
                p = pts[0]
                self.ax.text(p[0] + 5, p[1] + 5, p[2] + 5, str(order),
                             color="#00FF66", fontsize=8,
                             fontweight="bold", zorder=10)

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
        for rf in self._ref_frames:
            base_color = rf.get("color", "#FF88FF")
            origin = self._draw_frame_triad(
                rf["T"], 80, ["#FF4444", "#44FF44", "#4444FF"],
                axis_labels=["X", "Y", "Z"], lw=2.5, alpha=0.9)
            self.ax.scatter([origin[0]], [origin[1]], [origin[2]],
                            c=base_color, s=80, zorder=8, depthshade=False, marker="D")
            self.ax.text(origin[0]+12, origin[1]+12, origin[2]+12,
                         rf["name"], color=base_color, fontsize=7, fontweight="bold")

    # ── Cleanup ────────────────────────────────────────────────────────

    def destroy(self):
        plt.close(self.fig)
