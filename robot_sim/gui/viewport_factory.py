"""3D ビューポートのバックエンド選択（ストラングラー・パターン）。

このアプリの 3D ビューは長らく matplotlib(CPU) 版 `Viewport3D` を使ってきた。
高速化のため VisPy(OpenGL/GPU) 版 `ViewportGPU` への移行を進めているが、
移行中も既存版を壊さず、環境差（GPU/ドライバ無し）でも必ず起動できるよう、
ここでバックエンドを一元的に選択する。

選択方法（環境変数 ROBOT_VIEWPORT）:
    "mpl" (既定) … matplotlib(CPU) 版 Viewport3D
    "gpu"         … VisPy(OpenGL) 版 ViewportGPU
                    初期化に失敗した場合は自動で matplotlib 版にフォールバックする。

両バックエンドは同一の公開 API（update_robot / set_route / load_stl /
set_pick_curves / canvas_widget 等）を実装するため、呼び出し側
（main_window）はどちらが使われているかを意識しなくてよい。
"""
from __future__ import annotations

import os
import traceback
from typing import TYPE_CHECKING

import tkinter as tk

if TYPE_CHECKING:                       # 型チェック時のみ（実行時 import を避ける）
    from ..robot.kinematics import Kinematics


def create_viewport(parent: tk.Widget, kinematics: "Kinematics",
                    backend: str | None = None):
    """選択されたバックエンドの 3D ビューポートを生成して返す。

    backend を明示しない場合は環境変数 ROBOT_VIEWPORT（既定 "mpl"）に従う。
    "gpu" 指定時に VisPy 版の生成へ失敗したら matplotlib 版へ自動フォールバックし、
    どの環境でもアプリが起動できることを保証する。
    """
    backend = (backend or os.environ.get("ROBOT_VIEWPORT", "gpu")).strip().lower()

    if backend == "gpu":
        try:
            from .viewport_gpu import ViewportGPU
            vp = ViewportGPU(parent, kinematics)
            print("[viewport] GPU バックエンド（VisPy/OpenGL）を使用します。")
            return vp
        except Exception:
            print("[viewport] GPU バックエンドの初期化に失敗しました。"
                  " matplotlib 版にフォールバックします:")
            traceback.print_exc()

    # 既定 / フォールバック: matplotlib(CPU) 版
    from .viewport import Viewport3D
    return Viewport3D(parent, kinematics)
