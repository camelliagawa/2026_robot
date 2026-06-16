"""VisPy(OpenGL/GPU) 版 3D ビューポート — 段階移行 Phase 2。

matplotlib(CPU) 版 `Viewport3D` と同じ公開 API を実装し、GPU 描画により
回転・パン・ズーム・アニメーションを滑らかにする。低スペック PC でも
内蔵 GPU(Intel UHD 620) で 70fps 超を確認済み（PoC/埋め込み検証）。

移行ステータス:
  Phase 2 (このコミット): 埋め込み + ロボットメッシュ + カメラ + update_robot
  Phase 3: 静的シーン（床/作業領域/フレーム/STL/CSV/マーカー/ラベル）
  Phase 4: ルート表示 + 選択ハイライト
  Phase 5: ピッキング（クリック選択）
  Phase 6: アニメーション + 動画保存

Phase 2 では未実装の公開メソッドは「安全なスタブ」（no-op / 妥当な既定値）
として用意し、GPU バックエンドでもアプリがクラッシュせず起動・操作できる
ことを保証する。未実装機能（ルート線・STL 等の表示）は後続フェーズで追加する。
"""
from __future__ import annotations

import os
import struct
import tkinter as tk
from typing import List, Optional, TYPE_CHECKING

import numpy as np

# VisPy 本体（未インストール/初期化失敗時は factory が matplotlib にフォールバック）
from vispy import scene
from vispy.scene.visuals import Mesh

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
    ("base_link", (0.282, 0.301, 0.317, 1.0)),   # ベース: グレー
    ("link_1",    (0.960, 0.768, 0.000, 1.0)),   # J1〜J5: FANUC イエロー
    ("link_2",    (0.960, 0.768, 0.000, 1.0)),
    ("link_3",    (0.960, 0.768, 0.000, 1.0)),
    ("link_4",    (0.960, 0.768, 0.000, 1.0)),
    ("link_5",    (0.960, 0.768, 0.000, 1.0)),
    ("link_6",    (0.180, 0.180, 0.190, 1.0)),   # フランジ: ブラック
]


def _load_stl_tris(path: str):
    """バイナリ STL を読み込み (vertices(N*3,3) float32, faces(N,3) uint32) を返す。

    各三角形に固有の3頂点を割り当てる（flat shading 向け）。失敗時は None。
    """
    try:
        with open(path, "rb") as f:
            if len(f.read(80)) < 80:
                return None
            data = f.read(4)
            if len(data) < 4:
                return None
            (n_tri,) = struct.unpack("<I", data)
            buf = f.read(n_tri * 50)
        if len(buf) < n_tri * 50:
            return None
        raw  = np.frombuffer(buf, dtype=np.uint8).reshape(n_tri, 50)
        tris = raw[:, 12:48].view(np.float32).reshape(n_tri, 3, 3)
        verts = tris.reshape(-1, 3).astype(np.float32)
        faces = np.arange(n_tri * 3, dtype=np.uint32).reshape(n_tri, 3)
        return verts, faces
    except Exception:
        return None


def _urdf_link_transforms(q: np.ndarray) -> list:
    """各リンク(base_link, link_1..6)のワールド 4x4 変換を返す。

    viewport.py の _urdf_link_transforms と同一の規約（MDH→URDF 変換）。
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
        T = np.eye(4); T[:3, 3] = [x, y, z]; return T

    Ts = [np.eye(4)]                                       # base_link
    T = tr(0, 0, 330) @ rot("z", t1);      Ts.append(T)    # link_1
    T = T @ tr(50, 0, 0) @ rot("y", t2);   Ts.append(T)    # link_2
    T = T @ tr(0, 0, 440) @ rot("y", -t3); Ts.append(T)    # link_3
    T = T @ tr(0, 0, 35)  @ rot("x", -t4); Ts.append(T)    # link_4
    T = T @ tr(420, 0, 0) @ rot("y", -t5); Ts.append(T)    # link_5
    T = T @ tr(80, 0, 0)  @ rot("x", -t6); Ts.append(T)    # link_6
    return Ts


def _xform(verts: np.ndarray, T4: np.ndarray) -> np.ndarray:
    """(N,3) 頂点に 4x4 変換を適用。"""
    return (verts @ T4[:3, :3].T + T4[:3, 3]).astype(np.float32)


class _FigShim:
    """main_window が viewport.fig の図サイズ（ピクセル数）を参照するための互換シム。

    matplotlib 版は self.fig（Figure）を持つが GPU 版には無いため、
    動画/事前描画のフレーム枚数計算 (figwidth*dpi 等) が成立するよう
    キャンバスのピクセルサイズを返す最小オブジェクトを提供する。
    """
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
        self._fast_mode = False

        # ── 状態（後続フェーズで描画に使用）──────────────────────────────
        self._route: Optional["Route"] = None
        self._selected_wp_idx: Optional[int] = None
        self._tool_frame: Optional["ToolFrame"] = None
        self._user_frame: Optional["UserFrame"] = None
        self._jog_target: Optional[np.ndarray] = None
        self._ref_frames: list = []
        self._tcp_markers: list = []
        self._target_markers: list = []
        self._pick_curves: list = []
        self._pick_callback = None

        # ── main_window が直接参照する内部属性の互換用既定値 ──────────────
        # （いずれも「未読込/非再生」を表す安全な初期値。後続フェーズで本実装）
        self._stl_verts: Optional[np.ndarray] = None
        self._stl_T: np.ndarray = np.eye(4)
        self._blade_pts: Optional[np.ndarray] = None
        self._blade_normals: Optional[np.ndarray] = None
        self._pre_img = None   # 事前描画再生中フラグ（None=通常描画）

        # ── 実機リンクメッシュ（素の頂点・面・色）を読み込み ──────────────
        self._link_base = []   # [(verts(N*3,3), faces(N,3), rgba)]
        for name, rgba in _ROBOT_LINKS:
            res = _load_stl_tris(os.path.join(_ROBOT_MESH_DIR, name + ".stl"))
            if res is None:
                self._link_base = []   # 1つでも欠けたら全て無効（mpl 版と同じ方針）
                break
            self._link_base.append((res[0], res[1], rgba))

        # ── VisPy キャンバスを Tk フレームへ埋め込み ──────────────────────
        # app="tkinter" が pyopengltk 経由で Tk ウィジェットを生成する。
        # keys=None: 埋め込みでは VisPy 組込キー処理（Escで閉じる等）は不要。
        # "interactive" だと tkinter バックエンドで未対応キーの警告が出るため無効化。
        self.canvas = scene.SceneCanvas(
            parent=parent, app="tkinter",
            bgcolor="#161B22", keys=None)
        # main_window 側が pack / DnD バインドに使う Tk ウィジェット
        self.canvas_widget = self.canvas.native
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        # main_window が図サイズ参照に使う fig 互換シム
        self.fig = _FigShim(self.canvas)

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            elevation=22, azimuth=-50, distance=1700,
            center=(150, 0, 350), fov=35, up="+z")

        # ── ロボットメッシュ（リンクごとに Mesh を保持し set_data で更新）──
        self._link_meshes = []
        Ts = _urdf_link_transforms(self._joint_angles)
        for (verts, faces, rgba), T4 in zip(self._link_base, Ts):
            m = Mesh(vertices=_xform(verts, T4), faces=faces,
                     color=rgba, shading="flat", parent=self.view.scene)
            self._link_meshes.append(m)

        # 向き参照（簡易 XYZ 軸）— Phase 3 で本格的な床/フレームに置換予定
        scene.visuals.XYZAxis(parent=self.view.scene)

        self.update_robot(self._joint_angles)

    # ── ロボット姿勢（Phase 2 実装済み）────────────────────────────────
    def update_robot(self, joint_angles: np.ndarray):
        self._joint_angles = np.asarray(joint_angles, dtype=float)
        if not self._link_meshes:
            return
        Ts = _urdf_link_transforms(self._joint_angles)
        for m, (verts, faces, rgba), T4 in zip(
                self._link_meshes, self._link_base, Ts):
            m.set_data(vertices=_xform(verts, T4), faces=faces, color=rgba)
        self.canvas.update()

    def set_fast_mode(self, enabled: bool):
        # GPU 版では実機メッシュのまま十分滑らかなため軽量表示は不要。
        # 互換のためフラグのみ保持する（描画は常にフルメッシュ）。
        self._fast_mode = enabled

    def refresh(self):
        self.canvas.update()

    def destroy(self):
        try:
            self.canvas.close()
        except Exception:
            pass

    # ── 以下は後続フェーズで実装するスタブ（GPU 版が落ちないための最小実装）──

    # Phase 4: ルート / 選択
    def set_route(self, route: Optional["Route"]):
        self._route = route

    def set_selected_waypoint(self, idx: Optional[int]):
        self._selected_wp_idx = idx

    def set_jog_target(self, position: Optional[np.ndarray]):
        self._jog_target = position

    # Phase 3: フレーム / マーカー / オーバーレイ
    def set_tool_frame(self, tool_frame: Optional["ToolFrame"]):
        self._tool_frame = tool_frame

    def set_user_frame(self, user_frame: Optional["UserFrame"]):
        self._user_frame = user_frame

    def set_markers(self, tcp_markers: list, target_markers: list):
        self._tcp_markers = tcp_markers
        self._target_markers = target_markers

    def add_ref_frame(self, name, x, y, z, rx, ry, rz, color="#FF88FF"):
        self._ref_frames.append({"name": name})

    def remove_ref_frame(self, name: str):
        self._ref_frames = [r for r in self._ref_frames if r.get("name") != name]

    def clear_ref_frames(self):
        self._ref_frames = []

    def get_ref_frames(self) -> list:
        return list(self._ref_frames)

    # STL / CSV オーバーレイ（Phase 3）
    def load_stl(self, path: str):
        return None

    def set_stl_pose(self, x, y, z, rx, ry, rz):
        pass

    def clear_stl(self):
        pass

    def stl_bbox(self):
        return None

    def load_csv_points(self, path: str):
        return None

    def set_csv_pose(self, x, y, z, rx, ry, rz):
        pass

    def clear_csv(self):
        pass

    # 刃先 CSV（Phase 3）
    def load_blade_csv(self, path: str) -> int:
        return 0

    def set_blade_pose(self, x, y, z, rx, ry, rz):
        pass

    def clear_blade(self):
        pass

    def has_blade(self) -> bool:
        return False

    # レイヤースナップショット（Undo/Redo 用・Phase 3）
    def snapshot_layers(self) -> dict:
        return {}

    def restore_layers(self, snap: dict):
        pass

    # ピッキング（Phase 5）
    def set_pick_curves(self, curves: List[np.ndarray], callback,
                        local: bool = False):
        self._pick_curves = curves
        self._pick_callback = callback

    def set_pick_orders(self, orders: List[Optional[int]]):
        pass

    def clear_pick_curves(self):
        self._pick_curves = []
        self._pick_callback = None

    # 事前描画再生 / 動画（Phase 6）— GPU はリアルタイムで十分なため当面は簡易対応
    def render_frame(self, joint_angles) -> np.ndarray:
        self.update_robot(joint_angles)
        return self.canvas.render(alpha=True)

    def begin_prerendered_playback(self, first_frame: np.ndarray):
        pass

    def show_prerendered_frame(self, frame: np.ndarray):
        pass

    def end_prerendered_playback(self):
        pass
