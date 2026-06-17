"""3D ビューポートの生成（VisPy/OpenGL バックエンド）。

このアプリの 3D ビューは VisPy(OpenGL/GPU) 版 `ViewportGPU` を使用する。
かつては matplotlib(CPU) 版 `Viewport3D` を併用しストラングラー・パターンで
段階移行していたが、GPU 版が安定したため CPU 版は廃止し、GPU 単独構成にした。

GPU/OpenGL ドライバが利用できず初期化に失敗した場合は、原因が利用者に伝わる
エラーダイアログを表示してアプリを終了する（黙ってクラッシュさせない）。
"""
from __future__ import annotations

import sys
import traceback
from typing import TYPE_CHECKING

import tkinter as tk
from tkinter import messagebox

if TYPE_CHECKING:                       # 型チェック時のみ（実行時 import を避ける）
    from ..robot.kinematics import Kinematics


def create_viewport(parent: tk.Widget, kinematics: "Kinematics"):
    """VisPy(OpenGL) 版 3D ビューポートを生成して返す。

    初期化に失敗した場合は、GPU/OpenGL ドライバの問題である旨を示す
    エラーダイアログを表示し、SystemExit でアプリを終了する。
    """
    try:
        from .viewport_gpu import ViewportGPU
        vp = ViewportGPU(parent, kinematics)
        print("[viewport] GPU バックエンド（VisPy/OpenGL）を使用します。")
        return vp
    except Exception:
        print("[viewport] GPU バックエンド（VisPy/OpenGL）の初期化に失敗しました:")
        traceback.print_exc()
        detail = traceback.format_exc()
        try:
            messagebox.showerror(
                "3D ビューの初期化に失敗しました",
                "3D ビューポート（VisPy/OpenGL）の初期化に失敗しました。\n"
                "GPU / OpenGL ドライバが利用可能かご確認ください。\n\n"
                "必要なライブラリ:\n"
                "    pip install vispy pyopengl pyopengltk\n\n"
                "──── 詳細 ────\n"
                f"{detail}")
        except Exception:
            pass
        raise SystemExit(1)
