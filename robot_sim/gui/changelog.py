"""Version and changelog dialog for the FANUC LR Mate 200iD/14L simulator."""
import tkinter as tk
from tkinter import ttk

APP_VERSION = "0.3.0"

CHANGELOG = [
    ("0.3.0", "2026-06-04", [
        "ファナック LR Mate 200iD/14L の正式仕様を全面反映",
        "DH パラメータ修正: a1=50, a2=440, a3=35, d4=420mm (ROS-Industrial URDF 準拠)",
        "関節最大速度を追加: J1-J6 各軸の°/s 制限を管理",
        "ツールフレーム (UTool) 定義機能を追加",
        "ユーザーフレーム (UFrame) 定義機能を追加",
        "速度オーバーライド (Speed Override) スライダーを追加",
        "ジョグ操作 (Cartesian / Joint) パネルを追加",
        "3D ビューポート改善: フロアグリッド・ワークスペース可視化・TCP 表示",
        "チェンジログ画面を追加 (このダイアログ)",
        "ロボット仕様ダイアログを追加",
    ]),
    ("0.2.0", "2026-06-04", [
        "DH パラメータを Z-up 標準座標系に修正 (Y-up → Z-up)",
        "マウスホイールでの 3D ズーム機能を追加",
        "視点ドラッグ後の回転角度を保持するよう修正",
        "レディポジション: J2=-45° J3=+30° J5=-60° に設定",
        "デフォルト刃付けルートの座標系を統一",
    ]),
    ("0.1.0", "2026-06-04", [
        "初期実装: FANUC LR Mate 200iD/14L シミュレータ",
        "Modified DH 順・逆運動学 (解析解 + 数値フォールバック)",
        "matplotlib 3D ビューポート",
        "ウェイポイント編集 (追加/編集/削除)",
        "CSV インポート/エクスポート",
        "FANUC TP (.ls) プログラムエクスポート",
    ]),
]


def show_changelog(parent):
    """Show changelog dialog."""
    win = tk.Toplevel(parent)
    win.title(f"バージョン情報 / Changelog — v{APP_VERSION}")
    win.geometry("700x520")
    win.configure(bg="#1A1A1A")

    # Header
    hdr = tk.Label(
        win,
        text=f"FANUC LR Mate 200iD/14L  刃付けシミュレータ  v{APP_VERSION}",
        bg="#1A1A1A", fg="#F5C400", font=("", 12, "bold"),
    )
    hdr.pack(pady=(12, 4))
    sub = tk.Label(
        win, text="Knife Sharpening Robot Simulator",
        bg="#1A1A1A", fg="#888888", font=("", 9),
    )
    sub.pack()

    ttk.Separator(win).pack(fill=tk.X, padx=12, pady=8)

    # Scrollable text area
    frame = tk.Frame(win, bg="#1A1A1A")
    frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
    sb = tk.Scrollbar(frame)
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    txt = tk.Text(
        frame, bg="#111111", fg="#CCCCCC", font=("Courier", 9),
        yscrollcommand=sb.set, wrap=tk.WORD, state="normal",
        borderwidth=0, highlightthickness=0,
    )
    txt.pack(fill=tk.BOTH, expand=True)
    sb.config(command=txt.yview)

    for ver, date, items in CHANGELOG:
        txt.insert(tk.END, f"v{ver}  ({date})\n", "ver")
        for item in items:
            txt.insert(tk.END, f"  • {item}\n", "item")
        txt.insert(tk.END, "\n")

    txt.tag_config("ver", foreground="#F5C400", font=("Courier", 10, "bold"))
    txt.tag_config("item", foreground="#CCCCCC")
    txt.config(state="disabled")

    ttk.Button(win, text="閉じる (Close)", command=win.destroy).pack(pady=8)
