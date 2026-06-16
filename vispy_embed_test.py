#!/usr/bin/env python3
"""
Phase 0 — VisPy 埋め込み検証スパイク

目的: 「Tkinter ウィンドウの中に VisPy(OpenGL) の 3D ビューを埋め込み、
       隣に Tk のスライダー/ボタンを置く」がこのPCで動くかを確認する。
       これが動けば、3Dビューだけ VisPy に差し替え、残りのパネル類は
       Tkinter のまま、という安全な移行が可能になる。

== 初回セットアップ(コマンドプロンプトで1回だけ) ==
  pip install pyopengltk

  ※ vispy / pyopengl は導入済みのはず。pyopengltk は VisPy を
    Tkinter ウィジェットへ埋め込むために必要な追加パッケージ。

== 実行 ==
  cd C:\\Users\\koder\\2026_robot
  python vispy_embed_test.py

== 確認ポイント ==
  1. 左に 3D ロボット、右に Tk のコントロールが並んだ1枚のウィンドウが開く
  2. J2 / J3 スライダーを動かすとロボットがリアルタイムに動く
  3. 「視点リセット」ボタンが効く
  4. ウィンドウ下部に FPS が表示され、回転(左ドラッグ)が滑らか

  → すべて OK なら Phase 0 合格。本移行に進めます。
"""
import os
import sys
import struct
import tkinter as tk
from tkinter import ttk

import numpy as np

# ── 依存チェック ──────────────────────────────────────────────────────────
try:
    import pyopengltk  # noqa: F401  (VisPy tkinter バックエンドが内部で使用)
except ImportError:
    print("=" * 60)
    print("【エラー】pyopengltk がインストールされていません。")
    print("次を実行してから再起動してください:")
    print()
    print("  pip install pyopengltk")
    print("=" * 60)
    sys.exit(1)

try:
    from vispy import scene
    from vispy.scene.visuals import Mesh
except ImportError:
    print("=" * 60)
    print("【エラー】VisPy がインストールされていません:  pip install vispy pyopengl")
    print("=" * 60)
    sys.exit(1)

# ── パス設定 ─────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROBOT_DIR   = os.path.join(_HERE, "assets", "robot")
_TORMEK_PATH = os.path.join(_HERE, "assets", "Tormek_T8.stl")

_LINK_CONFIGS = [
    ("base_link", (0.282, 0.301, 0.317, 1.0)),
    ("link_1",    (0.960, 0.768, 0.000, 1.0)),
    ("link_2",    (0.960, 0.768, 0.000, 1.0)),
    ("link_3",    (0.960, 0.768, 0.000, 1.0)),
    ("link_4",    (0.960, 0.768, 0.000, 1.0)),
    ("link_5",    (0.960, 0.768, 0.000, 1.0)),
    ("link_6",    (0.180, 0.180, 0.190, 1.0)),
]
_TORMEK_POS = np.array([740.0, 240.0, 266.5], dtype=np.float32)


# ── STL ローダー ──────────────────────────────────────────────────────────
def _load_stl(path: str):
    """バイナリ STL を読み込み (vertices(N*3,3), faces(N,3)) を返す。"""
    with open(path, "rb") as f:
        f.read(80)
        (n_tri,) = struct.unpack("<I", f.read(4))
        buf = f.read(n_tri * 50)
    raw  = np.frombuffer(buf, dtype=np.uint8).reshape(n_tri, 50)
    tris = raw[:, 12:48].view(np.float32).reshape(n_tri, 3, 3)
    verts = tris.reshape(-1, 3).astype(np.float32)
    faces = np.arange(n_tri * 3, dtype=np.uint32).reshape(n_tri, 3)
    return verts, faces


# ── URDF FK (viewport.py の _urdf_link_transforms を移植) ─────────────────
def _urdf_transforms(q: np.ndarray) -> list:
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


# ── アプリ本体 ────────────────────────────────────────────────────────────
class EmbedTest:
    def __init__(self):
        self.q = np.deg2rad([0, -60, 10, 0, -30, 0])  # 初期姿勢

        # 各リンクの素(未変換)メッシュを読み込み
        self._link_base = []   # [(verts, faces, rgba)]
        for name, rgba in _LINK_CONFIGS:
            p = os.path.join(_ROBOT_DIR, name + ".stl")
            if os.path.isfile(p):
                v, f = _load_stl(p)
                self._link_base.append((v, f, rgba))
            else:
                print(f"  [skip] {p}")

        # ── Tk ルート: 左=3Dビュー / 右=コントロール ──────────────────────
        self.root = tk.Tk()
        self.root.title("Phase 0 — VisPy 埋め込み検証")
        self.root.geometry("1150x720")
        self.root.configure(bg="#161B22")

        left  = tk.Frame(self.root, bg="#161B22")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = tk.Frame(self.root, bg="#1C2333", width=240)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        # ── VisPy キャンバスを Tk フレームへ埋め込み ──────────────────────
        # app='tkinter' が pyopengltk 経由で Tk ウィジェットを生成する。
        self.canvas = scene.SceneCanvas(
            parent=left,
            app="tkinter",
            bgcolor="#161B22",
            keys="interactive",
        )
        # canvas.native が Tk ウィジェット。Tk レイアウトに pack する。
        self.canvas.native.pack(fill=tk.BOTH, expand=True)

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            elevation=20, azimuth=135, distance=1800,
            center=(0, 0, 400), fov=35, up="+z",
        )

        # ── ロボットメッシュ(リンクごとに Mesh を保持し set_data で更新) ──
        self._link_meshes = []
        Ts = _urdf_transforms(self.q)
        for (v, f, rgba), T4 in zip(self._link_base, Ts):
            m = Mesh(vertices=_xform(v, T4), faces=f,
                     color=rgba, shading="flat", parent=self.view.scene)
            self._link_meshes.append(m)

        # ── 砥石(固定) ──────────────────────────────────────────────────
        if os.path.isfile(_TORMEK_PATH):
            tv, tf = _load_stl(_TORMEK_PATH)
            Mesh(vertices=tv + _TORMEK_POS, faces=tf,
                 color=(0.45, 0.58, 0.75, 0.5), shading="flat",
                 parent=self.view.scene)

        scene.visuals.XYZAxis(parent=self.view.scene)

        # ── 右パネル: Tk コントロール ─────────────────────────────────────
        tk.Label(right, text="VisPy 埋め込みテスト", bg="#1C2333",
                 fg="#E6EDF3", font=("Yu Gothic UI", 12, "bold")).pack(pady=(16, 4))
        tk.Label(right, text="↓ スライダーでロボットが動けば\n   Tk⇄VisPy 連携 OK",
                 bg="#1C2333", fg="#88AACC",
                 font=("Yu Gothic UI", 9)).pack(pady=(0, 12))

        self._j2 = tk.DoubleVar(value=-60.0)
        self._j3 = tk.DoubleVar(value=10.0)
        self._add_slider(right, "J2 [deg]", self._j2, -120, 60)
        self._add_slider(right, "J3 [deg]", self._j3, -90, 90)

        ttk.Button(right, text="視点リセット",
                   command=self._reset_view).pack(pady=14, padx=20, fill=tk.X)

        self._status = tk.StringVar(value="FPS: --")
        tk.Label(right, textvariable=self._status, bg="#1C2333",
                 fg="#FFD400", font=("Consolas", 11, "bold")).pack(side=tk.BOTTOM, pady=16)

        # FPS をステータスへ
        self.canvas.measure_fps(window=0.5,
                                callback=lambda fps: self._status.set(f"FPS: {fps:.0f}"))

        print("=" * 56)
        print(" Phase 0 — VisPy 埋め込み検証")
        print("=" * 56)
        print(f" リンク数: {len(self._link_meshes)}  / 砥石STL: "
              f"{'あり' if os.path.isfile(_TORMEK_PATH) else 'なし'}")
        print(" → ウィンドウが開き、スライダーでロボットが動けば合格")
        print("=" * 56)

    def _add_slider(self, parent, label, var, lo, hi):
        tk.Label(parent, text=label, bg="#1C2333", fg="#E6EDF3",
                 font=("Yu Gothic UI", 9)).pack(pady=(8, 0))
        s = tk.Scale(parent, from_=lo, to=hi, orient=tk.HORIZONTAL,
                     variable=var, command=lambda e: self._update_robot(),
                     bg="#1C2333", fg="#E6EDF3", troughcolor="#0D1117",
                     highlightthickness=0, length=200, resolution=1.0)
        s.pack(padx=20, fill=tk.X)

    def _update_robot(self):
        """スライダー値からロボット姿勢を更新(リアルタイム)。"""
        self.q[1] = np.deg2rad(self._j2.get())
        self.q[2] = np.deg2rad(self._j3.get())
        Ts = _urdf_transforms(self.q)
        for m, (v, f, rgba), T4 in zip(self._link_meshes, self._link_base, Ts):
            m.set_data(vertices=_xform(v, T4), faces=f, color=rgba)
        self.canvas.update()

    def _reset_view(self):
        self.view.camera.elevation = 20
        self.view.camera.azimuth   = 135
        self.view.camera.distance  = 1800
        self.canvas.update()

    def run(self):
        # tkinter バックエンドでは Tk のメインループを回す
        self.root.mainloop()


if __name__ == "__main__":
    EmbedTest().run()
