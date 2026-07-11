"""VisPy(OpenGL/GPU) 版 3D ビューポート — 段階移行 Phase 3。

matplotlib(CPU) 版 `Viewport3D` と同じ公開 API を実装し、GPU 描画により
回転・パン・ズーム・アニメーションを滑らかにする。内蔵 GPU(Intel UHD 620)
でも 70fps 超を確認済み（PoC/埋め込み検証）。

移行ステータス:
  Phase 2: 埋め込み + ロボットメッシュ + カメラ + update_robot   ← 完了
  Phase 3 (このコミット): 静的シーン（床/作業領域/フレーム/STL/CSV/
          マーカー/ラベル）＋ ロボット追従要素（EEフレーム/ナイフ/刃先/TCP/影）
  Phase 4: ルート表示 + 選択ハイライト
  Phase 5: ピッキング（クリック選択）
  Phase 6: アニメーション最適化 + 動画保存

未実装（ルート線・選択・ピッキング・動画）は安全なスタブとして用意し、
GPU バックエンドでもクラッシュせず起動・操作できることを保証する。
"""
from __future__ import annotations

import os
import struct
import warnings
import tkinter as tk
from typing import List, Optional, TYPE_CHECKING

import numpy as np

from vispy import scene
from vispy.scene.visuals import Mesh, Line, Markers, Text

# VisPy tkinter バックエンドは未対応キー押下のたびに無害な UserWarning を出す
# （埋め込みでは VisPy のキー処理は使わない）。コンソールが埋まるため抑制する。
warnings.filterwarnings(
    "ignore",
    message="The key you typed is not supported by the tkinter backend.")

if TYPE_CHECKING:
    from ..robot.kinematics import Kinematics
    from ..path.route import Route
    from ..robot.tool_frame import ToolFrame
    from ..robot.user_frame import UserFrame


# ── 実機リンクメッシュ設定（viewport.py と同一）─────────────────────────
_ROBOT_MESH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "assets", "robot")
_ROBOT_LINKS = [
    ("base_link", (0.282, 0.301, 0.317, 1.0)),
    ("link_1",    (0.960, 0.768, 0.000, 1.0)),
    ("link_2",    (0.960, 0.768, 0.000, 1.0)),
    ("link_3",    (0.960, 0.768, 0.000, 1.0)),
    ("link_4",    (0.960, 0.768, 0.000, 1.0)),
    ("link_5",    (0.960, 0.768, 0.000, 1.0)),
    ("link_6",    (0.180, 0.180, 0.190, 1.0)),
]

# 軸トライアド色（X=赤 / Y=緑 / Z=青）
_AX_R = (1.00, 0.27, 0.27, 1.0)
_AX_G = (0.27, 1.00, 0.27, 1.0)
_AX_B = (0.27, 0.27, 1.00, 1.0)

KNIFE_BLADE_LEN   = 200.0
KNIFE_BLADE_WIDTH = 45.0

# 中ボタンドラッグでのパン速度倍率（1.0 = TurntableCamera 既定の Shift+左ドラッグと同じ）
_PAN_SPEED = 6.0

ROUTE_COLOR = "#00E5FF"   # ルート経路線（シアン）
WP_COLOR    = "#FF4422"   # 経路点（赤）
WP_ACTIVE   = "#00FF88"   # 選択中経路点（緑）
ROUTE_BIG_N = 25          # これを超えると per-point マーカー/ラベルを省略


def _rgba(hexstr: str, a: float = 1.0):
    h = hexstr.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0,
            int(h[4:6], 16) / 255.0, a)


def _load_stl_tris(path: str):
    """バイナリ/ASCII STL を (vertices(N*3,3) float32, faces(N,3) uint32) で返す。失敗時 None。"""
    try:
        with open(path, "rb") as f:
            if len(f.read(80)) < 80:
                return None
            data = f.read(4)
            if len(data) < 4:
                return None
            (n_tri,) = struct.unpack("<I", data)
            buf = f.read(n_tri * 50)
        if len(buf) == n_tri * 50 and n_tri > 0:
            raw  = np.frombuffer(buf, dtype=np.uint8).reshape(n_tri, 50)
            tris = raw[:, 12:48].view(np.float32).reshape(n_tri, 3, 3)
            verts = tris.reshape(-1, 3).astype(np.float32)
            faces = np.arange(n_tri * 3, dtype=np.uint32).reshape(n_tri, 3)
            return verts, faces
        # ASCII フォールバック
        vs = []
        with open(path, "r", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if s.startswith("vertex"):
                    p = s.split()
                    vs.append([float(p[1]), float(p[2]), float(p[3])])
        if len(vs) >= 3 and len(vs) % 3 == 0:
            verts = np.array(vs, dtype=np.float32)
            faces = np.arange(len(vs), dtype=np.uint32).reshape(-1, 3)
            return verts, faces
    except Exception:
        return None
    return None


def _urdf_link_transforms(q: np.ndarray) -> list:
    """各リンク(base_link, link_1..6)のワールド 4x4 変換（viewport.py と同一規約）。"""
    t1, t2, t3, t4, t5, t6 = (q[0], q[1] + np.pi / 2,
                               -q[2], -q[3], -q[4], -q[5])

    def rot(axis, t):
        c, s = np.cos(t), np.sin(t)
        T = np.eye(4)
        if axis == "z":
            T[:2, :2] = [[c, -s], [s, c]]
        elif axis == "y":
            T[0, 0], T[0, 2], T[2, 0], T[2, 2] = c, s, -s, c
        else:
            T[1, 1], T[1, 2], T[2, 1], T[2, 2] = c, -s, s, c
        return T

    def tr(x, y, z):
        T = np.eye(4); T[:3, 3] = [x, y, z]; return T

    Ts = [np.eye(4)]
    T = tr(0, 0, 330) @ rot("z", t1);      Ts.append(T)
    T = T @ tr(50, 0, 0) @ rot("y", t2);   Ts.append(T)
    T = T @ tr(0, 0, 440) @ rot("y", -t3); Ts.append(T)
    T = T @ tr(0, 0, 35)  @ rot("x", -t4); Ts.append(T)
    T = T @ tr(420, 0, 0) @ rot("y", -t5); Ts.append(T)
    T = T @ tr(80, 0, 0)  @ rot("x", -t6); Ts.append(T)
    return Ts


def _xform(verts: np.ndarray, T4: np.ndarray) -> np.ndarray:
    return (verts @ T4[:3, :3].T + T4[:3, 3]).astype(np.float32)


def _triad_segments(T: np.ndarray, scale: float):
    """4x4 変換から XYZ トライアドの線分 (pos(6,3), color(6,4)) を返す。"""
    o = T[:3, 3]
    R = T[:3, :3]
    pos = np.array([o, o + scale * R[:, 0],
                    o, o + scale * R[:, 1],
                    o, o + scale * R[:, 2]], dtype=np.float32)
    col = np.array([_AX_R, _AX_R, _AX_G, _AX_G, _AX_B, _AX_B], dtype=np.float32)
    return pos, col


class _FigShim:
    """main_window が viewport.fig のピクセルサイズを参照するための互換シム。"""
    dpi = 100.0

    def __init__(self, canvas):
        self._canvas = canvas

    def get_figwidth(self) -> float:
        return max(int(self._canvas.size[0]), 1) / self.dpi

    def get_figheight(self) -> float:
        return max(int(self._canvas.size[1]), 1) / self.dpi


class ViewportGPU:
    """VisPy(OpenGL) 製の 3D ビューポート（Viewport3D と同一 API）。"""

    def __init__(self, parent: tk.Widget, kinematics: "Kinematics"):
        self.kin = kinematics
        self._joint_angles = np.zeros(6)
        # GPU はリアルタイム描画が十分滑らかなため「事前描画再生」が不要。
        # main_window はこのフラグを見て滑らか再生をリアルタイム再生に切替える。
        self.realtime = True

        # ── レイヤー状態 ──────────────────────────────────────────────────
        self._route: Optional["Route"] = None
        self._selected_wp_idx: Optional[int] = None
        self._tool_frame: Optional["ToolFrame"] = None
        self._user_frame: Optional["UserFrame"] = None
        self._jog_target: Optional[np.ndarray] = None
        self._ref_frames: list = []
        self._tcp_markers: list = []
        self._target_markers: list = []

        self._stl_verts: Optional[np.ndarray] = None   # (N*3,3)
        self._stl_faces: Optional[np.ndarray] = None
        self._stl_name: str = ""
        self._stl_path: str = ""
        self._stl_T: np.ndarray = np.eye(4)

        self._csv_points: Optional[np.ndarray] = None  # (N,3)
        self._csv_name: str = ""
        self._csv_path: str = ""
        self._csv_T: np.ndarray = np.eye(4)

        self._blade_pts: Optional[np.ndarray] = None
        self._blade_normals: Optional[np.ndarray] = None
        self._blade_name: str = ""
        self._blade_path: str = ""
        self._blade_T: np.ndarray = np.eye(4)

        # ハンド取付ツール STL（フランジ追従・動的層）
        self._tool_verts: Optional[np.ndarray] = None
        self._tool_faces: Optional[np.ndarray] = None
        self._tool_name: str = ""
        self._tool_path: str = ""
        self._tool_T: np.ndarray = np.eye(4)

        self._pick_curves: list = []
        self._pick_curves_local = False
        self._pick_orders: list = []
        self._pick_callback = None

        # ── 実機リンクメッシュ（素データ）読み込み ────────────────────────
        self._link_base = []
        for name, rgba in _ROBOT_LINKS:
            res = _load_stl_tris(os.path.join(_ROBOT_MESH_DIR, name + ".stl"))
            if res is None:
                self._link_base = []
                break
            self._link_base.append((res[0], res[1], rgba))

        # ── VisPy キャンバスを Tk フレームへ埋め込み ──────────────────────
        self.canvas = scene.SceneCanvas(
            parent=parent, app="tkinter", bgcolor="#0D1117", keys=None)
        self.canvas_widget = self.canvas.native
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.fig = _FigShim(self.canvas)

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            elevation=22, azimuth=-50, distance=1700,
            center=(150, 0, 350), fov=35, up="+z")

        # 静的シーン用ノード（まとめて detach/再構築できるよう親ノードに集約）
        self._static_root = scene.Node(parent=self.view.scene)
        self._static_visuals: list = []
        # ルート/選択用ノード（ルート編集・選択変更時に再構築）
        self._route_root = scene.Node(parent=self.view.scene)
        self._route_visuals: list = []
        # 選択可能曲線用ノード（Phase 5 ピッキング）
        self._pick_root = scene.Node(parent=self.view.scene)
        self._pick_visuals: list = []
        self._pick_world: list = []   # 画面投影用のワールド座標曲線
        self._press_pos = None        # マウス押下位置（クリック/ドラッグ判定）

        # ── ロボットリンクメッシュ（持続）──────────────────────────────
        self._link_meshes = []
        Ts = _urdf_link_transforms(self._joint_angles)
        for (verts, faces, rgba), T4 in zip(self._link_base, Ts):
            m = Mesh(vertices=_xform(verts, T4), faces=faces,
                     color=rgba, shading="flat", parent=self.view.scene)
            self._link_meshes.append(m)

        # ── ロボット追従の動的ビジュアル（持続・set_data で更新）──────────
        self._ee_triad = Line(parent=self.view.scene, width=2.5,
                              connect="segments", antialias=True)
        self._shadow   = Line(parent=self.view.scene, width=3,
                              color=(0.2, 0.2, 0.2, 0.25))
        self._knife_lines = Line(parent=self.view.scene, width=4,
                                 connect="segments", antialias=True)
        self._knife_face  = Mesh(parent=self.view.scene)
        self._tool_mesh   = Mesh(parent=self.view.scene, shading="flat")
        self._blade_markers  = Markers(parent=self.view.scene)
        self._blade_whiskers = Line(parent=self.view.scene, width=1,
                                    connect="segments",
                                    color=_rgba("#FF99AA", 0.5))
        self._tcp_marker = Markers(parent=self.view.scene)
        self._tcp_line   = Line(parent=self.view.scene, width=1.5,
                                color=_rgba("#00FFCC", 0.9))
        self._tcp_label   = Text("", color="#00FFCC", font_size=8,
                                 anchor_x="left", parent=self.view.scene)
        self._blade_label = Text("", color="#FF7799", font_size=7,
                                 anchor_x="left", parent=self.view.scene)

        # クリックによる曲線ピッキング（TurntableCamera の回転と共存）
        # 中ボタンドラッグでのパン（TurntableCamera は既定で左=回転/右=ズームのみの
        # ため、中ボタンは未使用。左右・上下に視点を動かせるよう独自に実装する）
        self._pan_start = None   # (press_pos, camera.center) or None
        self.canvas.events.mouse_press.connect(self._on_mouse_press)
        self.canvas.events.mouse_move.connect(self._on_mouse_move)
        self.canvas.events.mouse_release.connect(self._on_mouse_release)

        self._rebuild_static()
        self.update_robot(self._joint_angles)

    # ────────────────────────────────────────────────────────────────────
    # 静的シーン
    # ────────────────────────────────────────────────────────────────────
    def _clear_static(self):
        for v in self._static_visuals:
            try:
                v.parent = None
            except Exception:
                pass
        self._static_visuals = []

    def _add_static(self, visual):
        visual.parent = self._static_root
        self._static_visuals.append(visual)
        return visual

    def _rebuild_static(self):
        """床/作業領域/フレーム/マーカー/STL/CSV/ジョグ目標を再構築する。"""
        self._clear_static()

        # ── 床グリッド（z=0・空間把握の参考）──────────────────────────
        seg = []
        for c in range(-700, 701, 100):
            seg += [[c, -700, 0], [c, 700, 0], [-700, c, 0], [700, c, 0]]
        self._add_static(Line(pos=np.array(seg, dtype=np.float32),
                              connect="segments", width=1,
                              color=_rgba("#21262D", 0.9)))

        # ── 作業領域円（肩高さ）─────────────────────────────────────────
        try:
            reach = float(self.kin.dh.REACH_MM)
            base_z = float(self.kin.dh.joints[0].d)
        except Exception:
            reach, base_z = 700.0, 330.0
        th = np.linspace(0, 2 * np.pi, 73)
        circ = np.column_stack([reach * np.cos(th), reach * np.sin(th),
                                np.full_like(th, base_z)]).astype(np.float32)
        self._add_static(Line(pos=circ, connect="strip", width=1,
                              color=_rgba("#1E3A5F", 0.5)))
        self._add_static(Text(f"{int(reach)}mm", pos=(reach * 0.72, 0, base_z + 30),
                             color=_rgba("#1E5A8F", 0.6), font_size=6))

        # ── ユーザーフレーム ──────────────────────────────────────────
        if self._user_frame is not None:
            T = self._user_frame.to_transform()
            pos, col = _triad_segments(T, 120)
            self._add_static(Line(pos=pos, color=col, connect="segments",
                                  width=2.5))
            o = T[:3, 3]
            self._add_static(Markers(pos=o[None, :], face_color="#FF88FF",
                                     size=10, edge_width=0, symbol="diamond"))
            nm = getattr(self._user_frame, "name", "UF")
            self._add_static(Text(f"[{nm}]", pos=(o[0] + 15, o[1] + 15, o[2] + 15),
                                  color="#FF88FF", font_size=8))

        # ── 参照フレーム ──────────────────────────────────────────────
        for rf in self._ref_frames:
            T = rf["T"]
            pos, col = _triad_segments(T, 90)
            self._add_static(Line(pos=pos, color=col, connect="segments", width=2))
            o = T[:3, 3]
            self._add_static(Text(rf.get("name", ""),
                                  pos=(o[0] + 12, o[1] + 12, o[2] + 12),
                                  color=rf.get("color", "#FF88FF"), font_size=7))

        # ── マーカー（TCP / ターゲット）─────────────────────────────────
        for m in self._tcp_markers:
            p = np.asarray(m["pos"], float)
            self._add_static(Markers(pos=p[None, :], face_color="#00FFCC",
                                     size=16, edge_width=0, symbol="star"))
            self._add_static(Text(f"[TCP] {m['name']}",
                                  pos=(p[0] + 14, p[1] + 14, p[2] + 14),
                                  color="#00FFCC", font_size=8, bold=True))
        for m in self._target_markers:
            p = np.asarray(m["pos"], float)
            self._add_static(Markers(pos=p[None, :], face_color="#FF8800",
                                     size=12, edge_width=0, symbol="disc"))
            self._add_static(Text(f"[TGT] {m['name']}",
                                  pos=(p[0] + 14, p[1] + 14, p[2] + 14),
                                  color="#FF8800", font_size=8, bold=True))

        # ── STL オーバーレイ（半透明）──────────────────────────────────
        if self._stl_verts is not None and self._stl_faces is not None:
            vw = _xform(self._stl_verts, self._stl_T)
            self._add_static(Mesh(vertices=vw, faces=self._stl_faces,
                                  color=(0.45, 0.58, 0.75, 0.45),
                                  shading="flat"))
            ctr = vw.mean(axis=0)
            zmax = float(vw[:, 2].max())
            self._add_static(Text(self._stl_name,
                                  pos=(ctr[0], ctr[1], zmax + 25),
                                  color="#99BBFF", font_size=7))

        # ── CSV 点群 ──────────────────────────────────────────────────
        if self._csv_points is not None:
            pw = _xform(self._csv_points, self._csv_T)
            self._add_static(Markers(pos=pw, face_color=_rgba("#FF9944", 0.7),
                                     size=6, edge_width=0, symbol="disc"))
            ctr = pw.mean(axis=0)
            self._add_static(Text(self._csv_name, pos=tuple(ctr),
                                  color="#FFBB66", font_size=6))

        # ── ジョグ目標 ──────────────────────────────────────────────────
        if self._jog_target is not None:
            x, y, z = self._jog_target
            s = 30
            cross = np.array([[x - s, y, z], [x + s, y, z],
                              [x, y - s, z], [x, y + s, z],
                              [x, y, z - s], [x, y, z + s]], dtype=np.float32)
            self._add_static(Line(pos=cross, connect="segments", width=1.5,
                                  color=_rgba("#44FF44", 0.9)))

        self.canvas.update()

    # ────────────────────────────────────────────────────────────────────
    # ロボット姿勢 + 追従要素
    # ────────────────────────────────────────────────────────────────────
    def update_robot(self, joint_angles: np.ndarray):
        self._joint_angles = np.asarray(joint_angles, dtype=float)
        q = self._joint_angles

        if self._link_meshes:
            Ts = _urdf_link_transforms(q)
            for m, (verts, faces, rgba), T4 in zip(
                    self._link_meshes, self._link_base, Ts):
                m.set_data(vertices=_xform(verts, T4), faces=faces, color=rgba)

        self._update_dynamic(q)
        # 刃先ローカルの選択曲線はフランジ姿勢に追従するため再構築
        if self._pick_curves and self._pick_curves_local:
            self._rebuild_pick()
        self.canvas.update()

    def _blade_axes(self, T_ee):
        T = T_ee @ self._blade_T
        o = T[:3, 3]; R = T[:3, :3]
        blade_dir = R[:, 1]; width_dir = R[:, 2]
        if self._blade_pts is not None and len(self._blade_pts):
            blade_len = float(np.max(self._blade_pts[:, 1]))
            if blade_len < 1.0:
                blade_len = KNIFE_BLADE_LEN
        else:
            blade_len = KNIFE_BLADE_LEN
        return o, blade_dir, width_dir, blade_len

    def _update_dynamic(self, q):
        try:
            T_ee = self.kin.forward(q)
        except Exception:
            return

        # EE フレームトライアド
        pos, col = _triad_segments(T_ee, 70)
        self._ee_triad.set_data(pos=pos, color=col)

        # 地面の影（関節原点の z=0 投影ポリライン）
        try:
            jp = self.kin.get_joint_positions(q)
            sp = jp.copy().astype(np.float32); sp[:, 2] = 0.0
            self._shadow.set_data(pos=sp)
            self._shadow.visible = True
        except Exception:
            self._shadow.visible = False

        # ナイフ（柄: フランジ→刃元 / 刃: 刃元→刃先）
        flange = T_ee[:3, 3]
        origin, blade_dir, width_dir, blade_len = self._blade_axes(T_ee)
        tip = origin + blade_len * blade_dir
        kpos = np.array([flange, origin, origin, tip], dtype=np.float32)
        kcol = np.array([_rgba("#3A2010"), _rgba("#3A2010"),
                         _rgba("#C8C8D0"), _rgba("#C8C8D0")], dtype=np.float32)
        self._knife_lines.set_data(pos=kpos, color=kcol)

        hw = KNIFE_BLADE_WIDTH / 2.0
        quad = np.array([origin - hw * width_dir, origin + hw * width_dir,
                         tip + hw * width_dir, tip - hw * width_dir],
                        dtype=np.float32)
        self._knife_face.set_data(vertices=quad,
                                  faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
                                  color=(0.78, 0.78, 0.82, 0.22))

        # ハンド取付ツール STL（フランジ追従）
        if self._tool_verts is not None and self._tool_faces is not None:
            T = T_ee @ self._tool_T
            self._tool_mesh.set_data(vertices=_xform(self._tool_verts, T),
                                     faces=self._tool_faces,
                                     color=(0.75, 0.76, 0.80, 1.0))
            self._tool_mesh.visible = True
        else:
            self._tool_mesh.visible = False

        # 刃先CSV点群（フランジ追従）
        if self._blade_pts is not None and len(self._blade_pts):
            T = T_ee @ self._blade_T
            R, t = T[:3, :3], T[:3, 3]
            pw = (self._blade_pts @ R.T + t).astype(np.float32)
            self._blade_markers.set_data(pos=pw, face_color=_rgba("#FF5577", 0.85),
                                         size=4, edge_width=0, symbol="disc")
            self._blade_markers.visible = True
            if self._blade_normals is not None:
                nw = (self._blade_normals @ R.T)
                p0 = pw[::8]; p1 = (pw[::8] + 8.0 * nw[::8]).astype(np.float32)
                seg = np.empty((len(p0) * 2, 3), dtype=np.float32)
                seg[0::2] = p0; seg[1::2] = p1
                self._blade_whiskers.set_data(pos=seg)
                self._blade_whiskers.visible = len(p0) > 0
            else:
                self._blade_whiskers.visible = False
            ctr = pw.mean(axis=0)
            self._blade_label.text = f"{self._blade_name} ({len(pw)} pts)"
            self._blade_label.pos = (ctr[0] + 10, ctr[1] + 10, ctr[2] + 10)
            self._blade_label.visible = True
        else:
            self._blade_markers.visible = False
            self._blade_whiskers.visible = False
            self._blade_label.visible = False

        # TCP マーカー（刃先端 or ツールフレーム）
        tcp_pos = None
        if self._blade_pts is not None and len(self._blade_pts):
            tcp_pos = origin + blade_len * blade_dir
        elif self._tool_frame is not None and getattr(self._tool_frame, "z", 0.0) != 0.0:
            tcp_pos = (T_ee @ self._tool_frame.to_transform())[:3, 3]
        if tcp_pos is not None:
            self._tcp_marker.set_data(pos=np.asarray(tcp_pos, float)[None, :],
                                      face_color="#00FFCC", size=14,
                                      edge_width=0, symbol="star")
            self._tcp_marker.visible = True
            self._tcp_line.set_data(pos=np.array([flange, tcp_pos], dtype=np.float32))
            self._tcp_line.visible = True
            self._tcp_label.text = "TCP"
            self._tcp_label.pos = (tcp_pos[0] + 8, tcp_pos[1] + 8, tcp_pos[2] + 8)
            self._tcp_label.visible = True
        else:
            self._tcp_marker.visible = False
            self._tcp_line.visible = False
            self._tcp_label.visible = False

    def refresh(self):
        self.canvas.update()

    def destroy(self):
        try:
            self.canvas.close()
        except Exception:
            pass

    # ────────────────────────────────────────────────────────────────────
    # フレーム / マーカー / オーバーレイ
    # ────────────────────────────────────────────────────────────────────
    def set_tool_frame(self, tool_frame: Optional["ToolFrame"]):
        self._tool_frame = tool_frame
        self.update_robot(self._joint_angles)

    def set_user_frame(self, user_frame: Optional["UserFrame"]):
        self._user_frame = user_frame
        self._rebuild_static()

    def set_jog_target(self, position: Optional[np.ndarray]):
        self._jog_target = position
        self._rebuild_static()

    def set_markers(self, tcp_markers: list, target_markers: list):
        self._tcp_markers = [
            {"name": m["name"], "pos": np.asarray(m["pos"], float)} for m in tcp_markers]
        self._target_markers = [
            {"name": m["name"], "pos": np.asarray(m["pos"], float)} for m in target_markers]
        self._rebuild_static()

    def add_ref_frame(self, name, x, y, z, rx, ry, rz, color="#FF88FF"):
        from ..robot.kinematics import Kinematics
        T = Kinematics.pose_to_transform(x, y, z, rx, ry, rz)
        self._ref_frames.append({"name": name, "T": T, "color": color})
        self._rebuild_static()

    def remove_ref_frame(self, name: str):
        self._ref_frames = [r for r in self._ref_frames if r.get("name") != name]
        self._rebuild_static()

    def clear_ref_frames(self):
        self._ref_frames = []
        self._rebuild_static()

    def get_ref_frames(self) -> list:
        return list(self._ref_frames)

    # STL
    def load_stl(self, path: str):
        res = _load_stl_tris(path)
        if res is None:
            return False
        self._stl_verts, self._stl_faces = res[0], res[1]
        self._stl_name = os.path.basename(path)
        self._stl_path = path
        self._rebuild_static()
        return True

    def set_stl_pose(self, x, y, z, rx, ry, rz):
        from ..robot.kinematics import Kinematics
        self._stl_T = Kinematics.pose_to_transform(x, y, z, rx, ry, rz)
        self._rebuild_static()

    def clear_stl(self):
        self._stl_verts = None; self._stl_faces = None
        self._stl_name = ""; self._stl_path = ""; self._stl_T = np.eye(4)
        self._rebuild_static()

    def stl_bbox(self):
        if self._stl_verts is None:
            return None
        v = self._stl_verts.reshape(-1, 3)
        return (v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max(),
                v[:, 2].min(), v[:, 2].max())

    # CSV
    def load_csv_points(self, path: str):
        import csv
        pts = []
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                for row in csv.reader(f):
                    if len(row) >= 3:
                        try:
                            pts.append([float(row[0]), float(row[1]), float(row[2])])
                        except ValueError:
                            pass
        except OSError:
            return False
        if not pts:
            return False
        self._csv_points = np.array(pts, dtype=float)
        self._csv_name = os.path.basename(path)
        self._csv_path = path
        self._rebuild_static()
        return True

    def set_csv_pose(self, x, y, z, rx, ry, rz):
        from ..robot.kinematics import Kinematics
        self._csv_T = Kinematics.pose_to_transform(x, y, z, rx, ry, rz)
        self._rebuild_static()

    def clear_csv(self):
        self._csv_points = None
        self._csv_name = ""; self._csv_path = ""; self._csv_T = np.eye(4)
        self._rebuild_static()

    # 刃先CSV（フランジ追従・動的層）
    def load_blade_csv(self, path: str) -> int:
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
                        pts.append(vals[:3]); nrm.append(vals[3:6])
        except OSError:
            return 0
        if not pts:
            return 0
        self._blade_pts = np.array(pts, dtype=float)
        na = np.array(nrm, dtype=float)
        lens = np.linalg.norm(na, axis=1, keepdims=True); lens[lens < 1e-9] = 1.0
        self._blade_normals = na / lens
        self._blade_name = os.path.basename(path)
        self._blade_path = path
        self.update_robot(self._joint_angles)
        return len(pts)

    def set_blade_pose(self, x, y, z, rx, ry, rz):
        from ..robot.kinematics import Kinematics
        self._blade_T = Kinematics.pose_to_transform(x, y, z, rx, ry, rz)
        self.update_robot(self._joint_angles)

    def clear_blade(self):
        self._blade_pts = None; self._blade_normals = None
        self._blade_name = ""; self._blade_path = ""; self._blade_T = np.eye(4)
        self.update_robot(self._joint_angles)

    def has_blade(self) -> bool:
        return self._blade_pts is not None

    # ハンド取付ツール STL（フランジ追従・動的層）
    def load_tool_stl(self, path: str) -> bool:
        res = _load_stl_tris(path)
        if res is None:
            return False
        self._tool_verts, self._tool_faces = res[0], res[1]
        self._tool_name = os.path.basename(path)
        self._tool_path = path
        self.update_robot(self._joint_angles)
        return True

    def set_tool_pose(self, x, y, z, rx, ry, rz):
        from ..robot.kinematics import Kinematics
        self._tool_T = Kinematics.pose_to_transform(x, y, z, rx, ry, rz)
        self.update_robot(self._joint_angles)

    def clear_tool_stl(self):
        self._tool_verts = None; self._tool_faces = None
        self._tool_name = ""; self._tool_path = ""; self._tool_T = np.eye(4)
        self.update_robot(self._joint_angles)

    def has_tool_stl(self) -> bool:
        return self._tool_verts is not None

    # レイヤースナップショット（Undo/Redo）
    _LAYER_FIELDS = (
        "_stl_verts", "_stl_faces", "_stl_name", "_stl_path", "_stl_T",
        "_csv_points", "_csv_name", "_csv_path", "_csv_T",
        "_blade_pts", "_blade_normals", "_blade_name", "_blade_path", "_blade_T",
        "_tool_verts", "_tool_faces", "_tool_name", "_tool_path", "_tool_T",
    )

    def snapshot_layers(self) -> dict:
        snap = {}
        for f in self._LAYER_FIELDS:
            v = getattr(self, f)
            snap[f] = v.copy() if f.endswith("_T") else v
        snap["ref_frames"] = [
            {"name": r["name"], "T": r["T"].copy(), "color": r.get("color", "#FF88FF")}
            for r in self._ref_frames]
        return snap

    def restore_layers(self, snap: dict):
        for f in self._LAYER_FIELDS:
            if f in snap:
                v = snap[f]
                setattr(self, f, v.copy() if (f.endswith("_T") and v is not None) else v)
        self._ref_frames = [
            {"name": r["name"], "T": r["T"].copy(), "color": r["color"]}
            for r in snap.get("ref_frames", [])]
        self._rebuild_static()
        self.update_robot(self._joint_angles)

    # ── ルート / 選択（Phase 4）──────────────────────────────────────────
    def set_route(self, route: Optional["Route"]):
        self._route = route
        self._rebuild_route()

    def set_selected_waypoint(self, idx: Optional[int]):
        self._selected_wp_idx = idx
        self._rebuild_route()

    def _clear_route(self):
        for v in self._route_visuals:
            try:
                v.parent = None
            except Exception:
                pass
        self._route_visuals = []

    def _add_route(self, visual):
        visual.parent = self._route_root
        self._route_visuals.append(visual)
        return visual

    def _rebuild_route(self):
        """ルート経路線・経路点マーカー・ラベル・選択ハイライトを再構築する。"""
        self._clear_route()
        route = self._route
        if route is None or len(route) == 0:
            self.canvas.update()
            return

        positions = np.asarray(route.positions_array(), dtype=np.float32)
        n = len(positions)

        # 経路線（ポリライン）
        if n >= 2:
            self._add_route(Line(pos=positions, connect="strip", width=2.5,
                                 color=_rgba(ROUTE_COLOR, 1.0), antialias=True))

        if n > ROUTE_BIG_N:
            # 軽量モード: 始点(緑)・終点(赤)のみ
            p0, p1 = positions[0], positions[-1]
            self._add_route(Markers(pos=p0[None, :], face_color=WP_ACTIVE,
                                    size=10, edge_width=0, symbol="disc"))
            self._add_route(Markers(pos=p1[None, :], face_color=WP_COLOR,
                                    size=10, edge_width=0, symbol="square"))
            self._add_route(Text(f"START ({n}点)",
                                 pos=(p0[0] + 10, p0[1] + 10, p0[2] + 10),
                                 color=WP_ACTIVE, font_size=6))
            self._add_route(Text("END", pos=(p1[0] + 10, p1[1] + 10, p1[2] + 10),
                                 color=WP_COLOR, font_size=6))
        else:
            # 全点を均一スタイルで描画
            self._add_route(Markers(pos=positions, face_color=WP_COLOR,
                                    size=8, edge_width=0, symbol="disc"))
            for i, wp in enumerate(route.waypoints):
                label = f"{i+1}:{wp.label}" if getattr(wp, "label", "") else f"P[{i+1}]"
                self._add_route(Text(label, pos=(wp.x + 10, wp.y + 10, wp.z + 10),
                                     color="#AAAAAA", font_size=6))

        # 選択ハイライト（緑の星 + 白ラベル）
        sel = self._selected_wp_idx
        if sel is not None and 0 <= sel < n:
            wp = route.waypoints[sel]
            self._add_route(Markers(pos=np.array([[wp.x, wp.y, wp.z]], dtype=np.float32),
                                    face_color=WP_ACTIVE, size=16,
                                    edge_width=0, symbol="star"))
            label = f"{sel+1}:{wp.label}" if getattr(wp, "label", "") else f"P[{sel+1}]"
            self._add_route(Text(label, pos=(wp.x + 10, wp.y + 10, wp.z + 10),
                                 color="white", font_size=7))

        self.canvas.update()

    # ── ピッキング（Phase 5）─────────────────────────────────────────────
    def set_pick_curves(self, curves: List[np.ndarray], callback,
                        *, blade_local: bool = False):
        self._pick_curves = [np.asarray(c, dtype=float) for c in curves]
        self._pick_curves_local = blade_local
        self._pick_orders = [None] * len(self._pick_curves)
        self._pick_callback = callback
        self._rebuild_pick()

    def set_pick_orders(self, orders: List[Optional[int]]):
        self._pick_orders = list(orders)
        self._rebuild_pick()

    def clear_pick_curves(self):
        self._pick_curves = []
        self._pick_curves_local = False
        self._pick_orders = []
        self._pick_callback = None
        self._rebuild_pick()

    def _clear_pick(self):
        for v in self._pick_visuals:
            try:
                v.parent = None
            except Exception:
                pass
        self._pick_visuals = []

    def _rebuild_pick(self):
        """選択可能曲線を描画する。未選択=シアン細線 / 選択=緑太線+順番号。

        刃先ローカル指定の場合は現在のフランジ姿勢でワールドへ変換し、
        画面投影によるクリック判定用に self._pick_world に保持する。
        """
        self._clear_pick()
        self._pick_world = []
        if not self._pick_curves:
            self.canvas.update()
            return

        if self._pick_curves_local:
            try:
                T = self.kin.forward(self._joint_angles) @ self._blade_T
            except Exception:
                T = self._blade_T
            Rw, tw = T[:3, :3], T[:3, 3]
            curves = [(c @ Rw.T + tw) for c in self._pick_curves]
        else:
            curves = self._pick_curves

        for i, pts in enumerate(curves):
            pts = np.asarray(pts, dtype=np.float32)
            self._pick_world.append(pts)
            order = self._pick_orders[i] if i < len(self._pick_orders) else None
            if order is not None:
                color, w = _rgba("#00FF66", 1.0), 3.0
            else:
                color, w = _rgba("#00CCDD", 0.9), 2.0
            v = Line(pos=pts, connect="strip", width=w, color=color,
                     antialias=True)
            v.parent = self._pick_root
            self._pick_visuals.append(v)
            if order is not None:
                p = pts[0]
                t = Text(str(order), pos=(p[0] + 5, p[1] + 5, p[2] + 5),
                         color="#00FF66", font_size=8, bold=True)
                t.parent = self._pick_root
                self._pick_visuals.append(t)

        self.canvas.update()

    def _pick_at(self, x: float, y: float) -> Optional[int]:
        """クリック画面座標 (x,y) に最も近い曲線の index を返す（閾値外は None）。"""
        if not self._pick_world:
            return None
        try:
            tr = self._ee_triad.get_transform("visual", "canvas")
        except Exception:
            return None
        best_i, best_d = None, 14.0   # 許容ピクセル距離
        for i, pts in enumerate(self._pick_world):
            proj = tr.map(pts)                       # (M,4)
            w = proj[:, 3:4]
            w[np.abs(w) < 1e-9] = 1e-9
            xy = proj[:, :2] / w
            d = np.sqrt((xy[:, 0] - x) ** 2 + (xy[:, 1] - y) ** 2).min()
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def _on_mouse_press(self, event):
        if event.button == 1:
            self._press_pos = np.asarray(event.pos, dtype=float)
        elif event.button == 3:
            cam = self.view.camera
            self._pan_start = (np.asarray(event.pos, dtype=float), tuple(cam.center))

    def _on_mouse_move(self, event):
        if self._pan_start is None or 3 not in event.buttons:
            return
        cam = self.view.camera
        p1, center0 = self._pan_start
        p2 = np.asarray(event.pos, dtype=float)
        # canvas.size は環境によって Tk のウィジェット座標系（マウス座標の単位）と
        # 一致しない場合があるため、実ウィジェットのピクセルサイズを直接使う。
        try:
            w = self.canvas_widget.winfo_width()
            h = self.canvas_widget.winfo_height()
        except Exception:
            w = h = 0
        if w <= 1 or h <= 1:
            w, h = self.canvas.size
        norm = float(np.mean((w, h)))
        if norm <= 0:
            return
        dist = (p1 - p2) / norm * cam.scale_factor * _PAN_SPEED
        dist[1] *= -1
        dx, dy, dz = cam._dist_to_trans(dist)
        up, forward, right = cam._get_dim_vectors()
        ff = cam._flip_factors
        dx, dy, dz = right * dx + forward * dy + up * dz
        dx, dy, dz = ff[0] * dx, ff[1] * dy, dz * ff[2]
        cam.center = (center0[0] + dx, center0[1] + dy, center0[2] + dz)

    def _on_mouse_release(self, event):
        if event.button == 3:
            self._pan_start = None
            return
        if event.button != 1 or self._press_pos is None:
            return
        rel = np.asarray(event.pos, dtype=float)
        moved = float(np.hypot(*(rel - self._press_pos)))
        self._press_pos = None
        # ドラッグ（回転）は無視。クリック（5px未満）のみピック判定。
        if moved >= 5.0 or self._pick_callback is None or not self._pick_world:
            return
        idx = self._pick_at(rel[0], rel[1])
        if idx is not None:
            self._pick_callback(idx)

    # ── 動画保存用フレーム描画 ──────────────────────────────────────────
    def render_frame(self, joint_angles) -> np.ndarray:
        self.update_robot(joint_angles)
        return self.canvas.render(alpha=True)
