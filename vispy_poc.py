#!/usr/bin/env python3
"""
VisPy GPU 描画 PoC — FANUC LR Mate 200iD/14L ロボットシミュレータ

matplotlib(CPU)版との回転スムーズさを比較するための最小検証スクリプト。
ロボット実機メッシュ(約8500三角形)＋砥石STL(約1800三角形)を OpenGL で描画する。

== 初回セットアップ(コマンドプロンプトで1回だけ実行) ==
  pip install vispy pyopengl

== 実行 ==
  python vispy_poc.py

== 操作 ==
  左ドラッグ   : 回転
  右ドラッグ   : ズーム
  中ドラッグ   : パン
  ホイール     : ズーム
  F キー       : 全画面トグル
  Q / Escape  : 終了
"""
import os
import sys
import struct
import numpy as np

# ── 依存チェック ──────────────────────────────────────────────────────────
try:
    from vispy import app, scene
    from vispy.scene.visuals import Mesh
except ImportError:
    print("=" * 60)
    print("【エラー】VisPy がインストールされていません。")
    print("以下のコマンドを実行してから再度起動してください:")
    print()
    print("  pip install vispy pyopengl")
    print("=" * 60)
    sys.exit(1)

# ── パス設定 ─────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROBOT_DIR   = os.path.join(_HERE, "assets", "robot")
_TORMEK_PATH = os.path.join(_HERE, "assets", "Tormek_T8.stl")

_LINK_CONFIGS = [
    ("base_link", (0.282, 0.301, 0.317, 1.0)),   # グレー
    ("link_1",    (0.960, 0.768, 0.000, 1.0)),   # FANUC イエロー
    ("link_2",    (0.960, 0.768, 0.000, 1.0)),
    ("link_3",    (0.960, 0.768, 0.000, 1.0)),
    ("link_4",    (0.960, 0.768, 0.000, 1.0)),
    ("link_5",    (0.960, 0.768, 0.000, 1.0)),
    ("link_6",    (0.180, 0.180, 0.190, 1.0)),   # ブラック
]

# 砥石STLの初期位置（matplotlib版のデフォルト値）
_TORMEK_POS = np.array([740.0, 240.0, 266.5], dtype=np.float32)


# ── STL ローダー ──────────────────────────────────────────────────────────
def _load_stl(path: str):
    """バイナリ STL を読み込み (vertices, faces, vertex_normals) を返す。

    vertices  : (N*3, 3) float32
    faces     : (N,   3) uint32   各面の頂点インデックス
    v_normals : (N*3, 3) float32  各頂点に面法線を割り当て(flat shading)
    """
    with open(path, "rb") as f:
        f.read(80)                                   # ヘッダスキップ
        (n_tri,) = struct.unpack("<I", f.read(4))
        buf = f.read(n_tri * 50)
    if len(buf) < n_tri * 50:
        raise ValueError(f"STL データが不完全です: {path}")

    raw = np.frombuffer(buf, dtype=np.uint8).reshape(n_tri, 50)
    face_normals = raw[:, :12 ].view(np.float32).reshape(n_tri, 3)   # (N,3)
    tris         = raw[:, 12:48].view(np.float32).reshape(n_tri, 3, 3) # (N,3,3)

    vertices  = tris.reshape(-1, 3).astype(np.float32)        # (N*3, 3)
    faces     = np.arange(n_tri * 3, dtype=np.uint32).reshape(n_tri, 3)
    v_normals = np.repeat(face_normals, 3, axis=0).astype(np.float32)  # (N*3, 3)
    return vertices, faces, v_normals


# ── URDF FK (viewport.py の _urdf_link_transforms をそのまま移植) ─────────
def _urdf_transforms(q: np.ndarray) -> list:
    """各リンクのワールド 4x4 変換リストを返す（q は 6関節角 [rad]）。"""
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

    Ts = [np.eye(4)]
    T = tr(0, 0, 330) @ rot("z", t1);    Ts.append(T)
    T = T @ tr(50, 0, 0) @ rot("y", t2); Ts.append(T)
    T = T @ tr(0, 0, 440) @ rot("y", -t3); Ts.append(T)
    T = T @ tr(0, 0, 35)  @ rot("x", -t4); Ts.append(T)
    T = T @ tr(420, 0, 0) @ rot("y", -t5); Ts.append(T)
    T = T @ tr(80, 0, 0)  @ rot("x", -t6); Ts.append(T)
    return Ts


def _apply_transform(verts: np.ndarray, T4: np.ndarray) -> np.ndarray:
    """(N,3) 頂点配列に 4x4 変換を適用して (N,3) を返す。"""
    R, t = T4[:3, :3], T4[:3, 3]
    return (verts @ R.T + t).astype(np.float32)


# ── メイン ────────────────────────────────────────────────────────────────
def main():
    # 表示姿勢: まあまあ見やすいレディーポジション
    # q = [0, -60°, 10°, 0, -30°, 0]
    q_ready = np.deg2rad([0, -60, 10, 0, -30, 0])
    transforms = _urdf_transforms(q_ready)

    # ── VisPy キャンバス ──────────────────────────────────────────────────
    canvas = scene.SceneCanvas(
        title="VisPy GPU PoC — FANUC LR Mate 200iD/14L  [Q: 終了]",
        keys="interactive",
        size=(1100, 750),
        show=True,
        bgcolor="#161B22",
    )
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(
        elevation=20,
        azimuth=135,
        distance=1800,
        center=(0, 0, 400),
        fov=35,
        up="+z",
    )

    total_tris = 0

    # ── ロボットリンクメッシュ ─────────────────────────────────────────────
    for (name, rgba), T4 in zip(_LINK_CONFIGS, transforms):
        path = os.path.join(_ROBOT_DIR, name + ".stl")
        if not os.path.isfile(path):
            print(f"  [skip] {path}")
            continue
        verts, faces, _ = _load_stl(path)
        verts_w = _apply_transform(verts, T4)
        # VisPy 0.16 の Mesh は法線を内部で自動計算する(shading="flat")。
        Mesh(
            vertices=verts_w,
            faces=faces,
            color=rgba,
            shading="flat",
            parent=view.scene,
        )
        total_tris += len(faces)

    # ── 砥石 STL オーバーレイ ─────────────────────────────────────────────
    if os.path.isfile(_TORMEK_PATH):
        verts, faces, _ = _load_stl(_TORMEK_PATH)
        verts_w = verts + _TORMEK_POS
        Mesh(
            vertices=verts_w,
            faces=faces,
            color=(0.45, 0.58, 0.75, 0.5),
            shading="flat",
            parent=view.scene,
        )
        total_tris += len(faces)
    else:
        print(f"  [skip] {_TORMEK_PATH}")

    # ── XYZ 軸（向きの参考・3Dシーン用の確実な表示）────────────────────
    scene.visuals.XYZAxis(parent=view.scene)

    # ── FPS 計測（ウィンドウタイトルに表示・最も確実な方法）──────────────
    base_title = f"VisPy GPU PoC — {total_tris:,} 三角形"

    def _show_fps(fps):
        canvas.title = f"{base_title} — {fps:.0f} FPS  [Q:終了]"

    # window 秒ごとに FPS を計測してタイトルへ反映
    canvas.measure_fps(window=0.5, callback=_show_fps)

    @canvas.events.key_press.connect
    def on_key(event):
        if event.key in ("q", "Q", "Escape"):
            app.quit()

    print()
    print("=" * 56)
    print(" VisPy GPU PoC — FANUC LR Mate 200iD/14L")
    print("=" * 56)
    print(f" 三角形数: {total_tris:,}  (matplotlib 版と同等)")
    print()
    print(" 操作:")
    print("   左ドラッグ   → 回転")
    print("   右ドラッグ   → ズーム")
    print("   中ドラッグ   → パン")
    print("   ホイール     → ズーム")
    print("   Q / Escape  → 終了")
    print()
    print(" → 回転がヌルヌル動くか確認してください")
    print("=" * 56)

    app.run()


if __name__ == "__main__":
    main()
