"""
Main tkinter window for the FANUC LR Mate 200iD/14L knife sharpening simulator.

Layout:
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Menu bar                                                             │
  ├────────────────────────────┬─────────────────────────────────────────┤
  │  3D Viewport               │  経路点リスト (Waypoint List)            │
  │  (matplotlib 3D)           │  追加/編集/削除/並べ替え                  │
  │                            │  Selected Waypoint Details               │
  │                            │  更新履歴パネル                           │
  ├────────────────────────────┴─────────────────────────────────────────┤
  │  関節角度スライダー J1-J6  │  速度OVR  │  UTool  │  UFrame            │
  ├──────────────────────────────────────────────────────────────────────┤
  │  ジョグ操作  │  ファイル  │  シミュレーション  │  IK  │  FK結果         │
  ├──────────────────────────────────────────────────────────────────────┤
  │  ステータスバー                                              [v0.3.1]  │
  └──────────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import concurrent.futures
import copy
import os
import time
import threading
import tkinter as tk
from contextlib import contextmanager
from dataclasses import replace
from tkinter import ttk, filedialog, messagebox, scrolledtext
from typing import Optional

import numpy as np

from ..robot.kinematics import Kinematics
from ..robot.tool_frame import ToolFrame
from ..robot.user_frame import UserFrame
from ..path.route import Route, Waypoint, MotionType
from ..path.csv_io import RouteCSVIO
from ..path.tp_exporter import TPExporter
from ..path.kenma_export import (
    generate_kenma_programs, export_kenma_ls, build_playback_sequence,
    parse_pose_expression, first_hover_T, enumerate_ik_branches,
    GenerationCancelled,
    load_blade_csv as load_blade_csv_file)
from .viewport_factory import create_viewport
from .route_editor import RouteEditor
from .changelog import show_changelog, APP_VERSION, CHANGELOG
from .undo import UndoManager
from . import settings_io

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD as _TkDnD
    _HAS_DND = True
except ImportError:
    _HAS_DND = False

# ── カラーパレット ──────────────────────────────────────────────────────
BG_DARK    = "#161B22"   # 最暗背景（GitHub dark風）
BG_PANEL   = "#21262D"   # パネル背景
BG_WIDGET  = "#2D333B"   # 入力欄・スライダー
BORDER     = "#444C56"   # 枠線
FG_PRIMARY = "#E6EDF3"   # 主テキスト（明）
FG_SUB     = "#8B949E"   # 補助テキスト（暗）
ACCENT     = "#F5C400"   # 強調色（黄）
ACCENT2    = "#58A6FF"   # アクセント2（青）
OK_GREEN   = "#3FB950"   # 成功色
ERR_RED    = "#F85149"   # エラー色
BTN_PRIMARY = "#1F6FEB"  # プライマリボタン
BTN_HOVER   = "#388BFD"


# ── パスユーティリティ ──────────────────────────────────────────────────

def _asset_path(name: str = "") -> str:
    """assets ディレクトリ（または配下のファイル）の絶対パスを返す。"""
    assets = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets")
    return os.path.join(assets, name) if name else assets


def _validate_path(path: str, ext: str, label: str):
    """realpath + isfile + 拡張子チェック。(正規化パス, エラーメッセージ or None) を返す。"""
    path = os.path.realpath(path)
    if not os.path.isfile(path):
        return path, "ファイルが見つかりません"
    if not path.lower().endswith(ext):
        return path, f"{label} ({ext}) を選択してください"
    return path, None


# ── ツールチップ ────────────────────────────────────────────────────────

# 表示メニュー「文字サイズ」(小/中/大) と連動する共通倍率。
# MainWindow._set_font_scale() が更新し、ツールチップは表示時に参照する。
_FONT_SCALE = 1.3
_TOOLTIP_BASE_PT = 9


class _Tooltip:
    """マウスホバーで説明を表示するツールチップ。

    フォントサイズは表示時に _FONT_SCALE（文字サイズ設定）から算出する
    ため、表示→文字サイズ の変更が既存ツールチップにも即時反映される。
    """
    def __init__(self, widget: tk.Widget, text: str):
        self._w    = widget
        self._text = text
        self._win  = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, event=None):
        if self._win:
            return
        x = self._w.winfo_rootx() + 16
        y = self._w.winfo_rooty() + self._w.winfo_height() + 6
        self._win = tw = tk.Toplevel(self._w)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        pt = max(7, int(round(_TOOLTIP_BASE_PT * _FONT_SCALE)))
        tk.Label(
            tw, text=self._text, justify="left",
            background="#1C2333", foreground="#E6EDF3",
            relief="solid", borderwidth=1,
            font=("Yu Gothic UI", pt),
            wraplength=max(340, int(340 * _FONT_SCALE)), padx=8, pady=5,
        ).pack()

    def _hide(self, event=None):
        if self._win:
            self._win.destroy()
            self._win = None


def _tip(widget: tk.Widget, text: str) -> _Tooltip:
    """ウィジェットにツールチップを設定して返す。"""
    return _Tooltip(widget, text)


class _NamedTarget:
    """Named TCP target / pose bookmark (RoboDK 風ターゲット)。

    RoboDK のターゲットと同様、ターゲット自体は「姿勢」のみを保持し、
    動作種別（MoveJ/MoveL）や速度は Route Sequence の命令側が持つ。
    """
    __slots__ = ("name", "x", "y", "z", "rx", "ry", "rz")

    def __init__(self, name: str, x: float = 0.0, y: float = 0.0,
                 z: float = 400.0, rx: float = 180.0, ry: float = 0.0, rz: float = 0.0):
        self.name = name
        self.x = float(x);  self.y = float(y);  self.z = float(z)
        self.rx = float(rx); self.ry = float(ry); self.rz = float(rz)

    def update_pose(self, x, y, z, rx, ry, rz):
        """Teach: 姿勢を現在値で上書きする。"""
        self.x = float(x);  self.y = float(y);  self.z = float(z)
        self.rx = float(rx); self.ry = float(ry); self.rz = float(rz)

    def as_waypoint(self, speed: float = 50.0,
                    motion: MotionType = MotionType.JOINT) -> Waypoint:
        return Waypoint(x=self.x, y=self.y, z=self.z,
                        rx=self.rx, ry=self.ry, rz=self.rz,
                        speed=speed, motion_type=motion, label=self.name)


class MainWindow:
    """Top-level application window."""

    APP_TITLE = "FANUC LR Mate 200iD/14L  ｜  刃付けロボットシミュレータ"
    MIN_WIDTH  = 1560
    MIN_HEIGHT = 860

    TOOL_FRAMES  = [ToolFrame.flange(), ToolFrame.default_knife()]
    # UF0 WORLD + UF9 STONE のみ（旧 UF1 STONE は UF9 と重複のため統一済み）
    USER_FRAMES  = [UserFrame.world(), UserFrame.stone9()]

    def __init__(self):
        self.kin   = Kinematics()
        self.route = Route.default_sharpening_route()
        self._joint_angles  = self.kin.dh.ready_position().copy()
        self._sim_thread: Optional[threading.Thread] = None
        self._sim_running   = False
        self._active_tool   = self.TOOL_FRAMES[1]
        self._active_uframe = self.USER_FRAMES[0]
        self._tree_programs: list = []   # [(prog_name, Route)]
        self._blade_csv_path: Optional[str] = None
        self._named_targets: list = []   # List[_NamedTarget]
        self._route_sequence: list = []  # List[dict] — {"type":"route"|"target", ...}
        self._seq_dirty = True           # True when sequence changed since last apply to route

        # ── Undo/Redo（全機能やり直し対応） ──
        self._suppress_undo = False   # 再生・復元中の自動更新は記録しない
        self._undo_mgr = UndoManager(self._capture_state, self._restore_state)

        # 曲線選択ダイアログの保持状態（オフセット式・選択順・逆方向）。
        # ダイアログを閉じても保持し、開き直して何度でもやり直せる。
        self._kenma_dlg_state: dict = {
            "offset": "rotx(0)*roty(0)*rotz(0)", "sel": [], "rev": {}}

        # ── シークバー / IK 事前計算状態 ──
        self._sim_solutions: list = []        # List[np.ndarray] 各経路点の関節解
        self._sim_solutions_ready = False     # 事前計算完了フラグ
        self._pre_frames = None               # 案1: 事前描画フレーム（RGBA画像）
        self._pre_idxs = None                 # 各フレームの分数インデックス
        self._pre_fps = 20.0                  # フレームレート（動画保存でも使用）
        self._sim_precompute_thread: Optional[threading.Thread] = None
        self._sim_precompute_token = 0        # ルート変更ごとに増分（古い結果を破棄）
        self._seek_dragging = False           # スクラブ中フラグ
        self._seek_updating = False           # スライダー自動更新中フラグ（コールバック抑制）
        self._sim_playing = False             # 再生中フラグ

        # 文字サイズ（小/中/大）— 既定は「中」
        self._orig_fonts: dict = {}
        self._font_scale: float = 1.3

        self._build_root()
        self._build_menu()
        # Bottom panels must be packed BEFORE expand=True main panel
        # so they always claim their space regardless of window height
        self._build_status_bar()
        self._build_main_panels()

        self.viewport.set_route(self.route)
        self.viewport.set_tool_frame(self._active_tool)
        self.viewport.set_user_frame(self._active_uframe)
        self._update_viewport_from_angles(self._joint_angles)
        self._update_fk_display()

        # 研磨機（Tormek T8 STL）を起動時から表示（研削経路CSVは読み込まない）
        # 起動時の初期化は undo 履歴に積まない
        self._suppress_undo = True
        try:
            self._load_tormek_stl()
        except Exception:
            pass
        finally:
            self._suppress_undo = False

        # 既定の文字サイズ（中）を全体に適用
        self._set_font_scale(self._font_scale)

        # シークバー用 IK 事前計算（バックグラウンド）
        self._invalidate_sim_solutions()

    # ──────────────────────────────────────────────────────────────────
    # Root window & style
    # ──────────────────────────────────────────────────────────────────

    def _build_root(self):
        if _HAS_DND:
            self.root = _TkDnD.Tk()
        else:
            self.root = tk.Tk()
        self.root.title(self.APP_TITLE)
        self.root.minsize(self.MIN_WIDTH, self.MIN_HEIGHT)
        self.root.configure(bg=BG_DARK)

        s = ttk.Style()
        s.theme_use("clam")

        # 基本設定
        s.configure(".",
            background=BG_PANEL, foreground=FG_PRIMARY,
            fieldbackground=BG_WIDGET, bordercolor=BORDER,
            troughcolor=BG_WIDGET, selectbackground=BTN_PRIMARY,
            selectforeground=FG_PRIMARY, font=("Yu Gothic UI", 9))

        # フレーム・ラベル
        s.configure("TFrame",      background=BG_PANEL)
        s.configure("TLabel",      background=BG_PANEL, foreground=FG_PRIMARY)
        s.configure("TLabelframe", background=BG_PANEL, foreground=FG_SUB,
                    bordercolor=BORDER, relief="flat")
        s.configure("TLabelframe.Label",
                    background=BG_PANEL, foreground=ACCENT2,
                    font=("Yu Gothic UI", 9, "bold"))

        # ボタン
        s.configure("TButton",
            padding=(8, 4), relief="flat",
            background=BG_WIDGET, foreground=FG_PRIMARY,
            bordercolor=BORDER, focuscolor=BG_WIDGET)
        s.map("TButton",
            background=[("active", "#3D444D"), ("pressed", BTN_PRIMARY)],
            foreground=[("active", FG_PRIMARY)])

        # プライマリボタン（実行系）
        s.configure("Primary.TButton",
            padding=(8, 4), relief="flat",
            background=BTN_PRIMARY, foreground="white",
            bordercolor=BTN_PRIMARY, font=("Yu Gothic UI", 9, "bold"))
        s.map("Primary.TButton",
            background=[("active", BTN_HOVER), ("pressed", "#1A5CC8")])

        # 危険ボタン（停止）
        s.configure("Danger.TButton",
            padding=(8, 4), relief="flat",
            background="#3D1E1E", foreground=ERR_RED,
            bordercolor="#6E2222")
        s.map("Danger.TButton",
            background=[("active", "#5A2020")])

        # ジョグボタン
        s.configure("Jog.TButton",
            padding=(2, 3), relief="flat", width=3,
            background="#2A3A4A", foreground=ACCENT2,
            bordercolor="#3D5A6E", font=("", 10, "bold"))
        s.map("Jog.TButton",
            background=[("active", "#3D5A6E")])

        # 入力欄
        s.configure("TEntry",    fieldbackground=BG_WIDGET, foreground=FG_PRIMARY,
                    bordercolor=BORDER, insertcolor=FG_PRIMARY)
        s.configure("TCombobox", fieldbackground=BG_WIDGET, foreground=FG_PRIMARY,
                    selectbackground=BTN_PRIMARY, arrowcolor=FG_SUB)
        s.map("TCombobox", fieldbackground=[("readonly", BG_WIDGET)])

        # スライダー
        s.configure("TScale", background=BG_PANEL, troughcolor=BG_WIDGET,
                    sliderlength=14, sliderrelief="flat")

        # Spinbox
        s.configure("TSpinbox", fieldbackground=BG_WIDGET, foreground=FG_PRIMARY,
                    arrowcolor=FG_SUB, bordercolor=BORDER)

        # Treeview（ルートエディタ）
        s.configure("Treeview",
            background=BG_WIDGET, foreground=FG_PRIMARY,
            fieldbackground=BG_WIDGET, rowheight=20,
            bordercolor=BORDER)
        s.configure("Treeview.Heading",
            background=BG_PANEL, foreground=ACCENT2,
            relief="flat", font=("Yu Gothic UI", 8, "bold"))
        s.map("Treeview",
            background=[("selected", BTN_PRIMARY)],
            foreground=[("selected", "white")])

        # セパレータ
        s.configure("TSeparator", background=BORDER)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ──────────────────────────────────────────────────────────────────
    # 文字サイズ（小/中/大）
    # ──────────────────────────────────────────────────────────────────

    def _fnt(self, size: int, *flags, fam: str = "Yu Gothic UI"):
        """現在の文字サイズ倍率を適用したフォントタプルを返す（後生成ダイアログ用）。"""
        return (fam, max(6, int(round(size * self._font_scale))), *flags)

    def _set_font_scale(self, scale: float):
        """UI全体の文字サイズを倍率 scale で再設定する（小=1.1/中=1.3/大=1.6）。"""
        import tkinter.font as tkfont
        global _FONT_SCALE
        _FONT_SCALE = scale       # ツールチップが表示時に参照する共通倍率
        self._font_scale = scale

        def fnt(size, *flags, fam="Yu Gothic UI"):
            return (fam, max(6, int(round(size * scale))), *flags)

        # ── ttk スタイルのフォントを再設定 ──────────────────────────
        s = ttk.Style()
        s.configure(".",                  font=fnt(9))
        s.configure("TButton",            font=fnt(9))
        s.configure("Primary.TButton",    font=fnt(9, "bold"))
        s.configure("Danger.TButton",     font=fnt(9))
        s.configure("Jog.TButton",        font=fnt(10, "bold", fam=""))
        s.configure("TLabelframe.Label",  font=fnt(9, "bold"))
        s.configure("TEntry",             font=fnt(9))
        s.configure("TCombobox",          font=fnt(9))
        s.configure("Treeview",           font=fnt(9),
                    rowheight=max(16, int(round(20 * scale))))
        s.configure("Treeview.Heading",   font=fnt(8, "bold"))

        # ── tk ウィジェット（Label/Button/Entry/Text/Listbox）を再設定 ──
        def walk(w):
            try:
                cur = w.cget("font")
            except Exception:
                cur = ""
            if cur:
                key = str(w)
                if key not in self._orig_fonts:
                    self._orig_fonts[key] = cur
                try:
                    fo = tkfont.Font(font=self._orig_fonts[key])
                    fam    = fo.actual("family")
                    size   = abs(fo.actual("size"))
                    flags  = []
                    if fo.actual("weight") == "bold":
                        flags.append("bold")
                    if fo.actual("slant") == "italic":
                        flags.append("italic")
                    w.configure(font=(fam, max(6, int(round(size * scale))), *flags))
                except Exception:
                    pass
            for c in w.winfo_children():
                walk(c)

        walk(self.root)

    # ──────────────────────────────────────────────────────────────────
    # Menu bar
    # ──────────────────────────────────────────────────────────────────

    def _build_menu(self):
        menubar = tk.Menu(self.root, bg=BG_PANEL, fg=FG_PRIMARY,
                          activebackground=BTN_PRIMARY, activeforeground="white",
                          borderwidth=0, relief="flat")

        def menu(label):
            m = tk.Menu(menubar, tearoff=0, bg=BG_PANEL, fg=FG_PRIMARY,
                        activebackground=BTN_PRIMARY, activeforeground="white",
                        borderwidth=1, relief="solid")
            menubar.add_cascade(label=label, menu=m)
            return m

        # ファイル
        f = menu("  ファイル (File)  ")
        f.add_command(label="  📂  CSV を開く...          Ctrl+O", command=self._load_csv)
        f.add_command(label="  💾  CSV として保存...      Ctrl+S", command=self._save_csv)
        f.add_separator()
        f.add_command(label="  📤  FANUC TP 出力...       Ctrl+E", command=self._export_tp)
        f.add_separator()
        f.add_command(label="  ⚙   設定をファイルへ出力...", command=self._export_settings)
        f.add_command(label="  📥  設定ファイルを読み込む...", command=self._import_settings)
        f.add_separator()
        f.add_command(label="  ✕   終了", command=self._on_close)

        # 編集（Undo/Redo — 全機能やり直し対応）
        e = menu("  編集 (Edit)  ")
        e.add_command(label="  ↩  元に戻す              Ctrl+Z", command=self._undo)
        e.add_command(label="  ↪  やり直す              Ctrl+Y", command=self._redo)
        e.configure(postcommand=self._update_edit_menu)
        self._edit_menu = e

        # ルート
        r = menu("  ルート (Route)  ")
        r.add_command(label="  🔪  研磨経路CSVを読み込む (kenma形式)", command=self._load_kenma_route)
        r.add_command(label="  📐  kenma形式LS出力（3ファイル1組）...",   command=self._export_kenma_ls)
        r.add_command(label="  📐  曲線を選択して研磨ルート生成...",       command=self._kenma_curve_select_dialog)
        r.add_command(label="  🗑   ルートをクリア",                    command=self._clear_route)
        r.add_separator()
        r.add_command(label="  🪨  Tormek T8 砥石を3D表示（STLのみ）",    command=self._load_tormek_stl)
        r.add_command(label="  📈  Tormek 研削経路CSVを表示",              command=self._load_tormek_csv)
        r.add_separator()
        r.add_command(label="  ▶   シミュレーション実行      F5",       command=self._start_simulation)

        # ロボット
        rb = menu("  ロボット (Robot)  ")
        rb.add_command(label="  🏠  ホームポジションへ移動",   command=self._go_home)
        rb.add_command(label="  🦾  レディポジションへ移動",   command=self._go_ready)
        rb.add_separator()
        rb.add_command(label="  🔧  ツールフレーム (UTool) 編集...", command=self._edit_tool_frame)
        rb.add_command(label="  📐  ユーザーフレーム (UFrame) 編集...", command=self._edit_user_frame)
        rb.add_separator()
        rb.add_command(label="  📊  DH パラメータを表示",     command=self._show_dh_params)
        rb.add_command(label="  📋  ロボット仕様を表示",       command=self._show_robot_specs)

        # 表示
        v = menu("  表示 (View)  ")
        self._font_size_var = tk.StringVar(value="中")
        for lbl, sc in [("小", 1.1), ("中", 1.3), ("大", 1.6)]:
            v.add_radiobutton(
                label=f"  🔠  文字サイズ: {lbl}",
                variable=self._font_size_var, value=lbl,
                command=lambda s=sc: self._set_font_scale(s))

        # ヘルプ
        h = menu("  ヘルプ (Help)  ")
        h.add_command(label=f"  📝  更新履歴 (v{APP_VERSION})...", command=lambda: show_changelog(self.root))
        h.add_command(label="  ℹ   About",                        command=self._show_about)

        self.root.config(menu=menubar)
        self.root.bind("<Control-o>", lambda e: self._load_csv())
        self.root.bind("<Control-s>", lambda e: self._save_csv())
        self.root.bind("<Control-e>", lambda e: self._export_tp())
        self.root.bind("<F5>",        lambda e: self._start_simulation())
        self.root.bind("<Control-z>", lambda e: self._undo())
        self.root.bind("<Control-y>", lambda e: self._redo())
        self.root.bind("<Control-Z>", lambda e: self._redo())   # Ctrl+Shift+Z

    # ──────────────────────────────────────────────────────────────────
    # Undo / Redo（スナップショット方式 — 全機能やり直し対応）
    # ──────────────────────────────────────────────────────────────────

    def _push_undo(self, label: str, coalesce: bool = False):
        """状態を変更する直前に呼ぶ。coalesce=True で連続操作をまとめる。"""
        if self._suppress_undo:
            return
        self._undo_mgr.push(label, coalesce=coalesce)

    @contextmanager
    def _undo_group(self, label: str):
        """複数の変更を 1 回の undo 単位にまとめるコンテキスト。

        内側で呼ばれる _push_undo は記録されない（二重記録防止）。
        """
        self._push_undo(label)
        prev = self._suppress_undo
        self._suppress_undo = True
        try:
            yield
        finally:
            self._suppress_undo = prev

    def _undo(self):
        if self._sim_running:
            self._stop_simulation()
        label = self._undo_mgr.undo()
        if label:
            self._set_status(f"↩  元に戻す: {label}")
        else:
            self._set_status("元に戻せる操作はありません")
        return "break"

    def _redo(self):
        if self._sim_running:
            self._stop_simulation()
        label = self._undo_mgr.redo()
        if label:
            self._set_status(f"↪  やり直す: {label}")
        else:
            self._set_status("やり直せる操作はありません")
        return "break"

    def _update_edit_menu(self):
        """編集メニューを開くたびに Undo/Redo ラベルと有効状態を更新する。"""
        ul = self._undo_mgr.undo_label
        rl = self._undo_mgr.redo_label
        self._edit_menu.entryconfig(
            0, label=f"  ↩  元に戻す{f': {ul}' if ul else ''}              Ctrl+Z",
            state="normal" if ul else "disabled")
        self._edit_menu.entryconfig(
            1, label=f"  ↪  やり直す{f': {rl}' if rl else ''}              Ctrl+Y",
            state="normal" if rl else "disabled")

    def _capture_state(self) -> dict:
        """Undo 用の全状態スナップショットを返す（操作直前に呼ばれる）。"""
        return {
            "route": (self.route.name, self.route.comment,
                      self.route.uframe, self.route.utool,
                      copy.deepcopy(self.route.waypoints)),
            "markers": (copy.deepcopy(self._mk_list),
                        self._mk_tcp_count, self._mk_tgt_count),
            "tool_frames": [replace(t) for t in self.TOOL_FRAMES],
            "user_frames": [replace(u) for u in self.USER_FRAMES],
            "active_tool_num": self._active_tool.number,
            "active_uframe_num": self._active_uframe.number,
            "joint_angles": self._joint_angles.copy(),
            "stl_vars":   [v.get() for v in self._stl_pose_vars],
            "csv_vars":   [v.get() for v in self._csv_pose_vars],
            "blade_vars": [v.get() for v in self._blade_pose_vars],
            "stone_vars": [v.get() for v in self._stone_vars],
            "blade_csv_path": self._blade_csv_path,
            "tree_programs": list(self._tree_programs),
            "viewport": self.viewport.snapshot_layers(),
            "kenma_dlg": copy.deepcopy(self._kenma_dlg_state),
            "speed_override": self._speed_override.get(),
        }

    def _restore_state(self, s: dict):
        """スナップショットの状態へ戻し、全 UI を同期する。"""
        prev = self._suppress_undo
        self._suppress_undo = True
        try:
            # 経路
            (self.route.name, self.route.comment,
             self.route.uframe, self.route.utool, wps) = s["route"]
            self.route.waypoints = wps
            self.route_editor.set_route(self.route)

            # マーカー
            self._mk_list, self._mk_tcp_count, self._mk_tgt_count = s["markers"]
            self._mk_refresh_listbox()
            self._mk_sync_viewport()

            # フレーム定義 + アクティブ UTool/UFrame
            self.TOOL_FRAMES[:] = s["tool_frames"]
            self.USER_FRAMES[:] = s["user_frames"]
            self._utool_combo.config(values=[
                f"UT{t.number}: {t.name}  (z={t.z:.0f}mm)"
                for t in self.TOOL_FRAMES])
            self._uframe_combo.config(values=[
                f"UF{u.number}: {u.name}" for u in self.USER_FRAMES])
            ti = next((i for i, t in enumerate(self.TOOL_FRAMES)
                       if t.number == s["active_tool_num"]), 0)
            self._utool_combo.current(ti)
            self._active_tool = self.TOOL_FRAMES[ti]
            self.viewport.set_tool_frame(self._active_tool)
            ui = next((i for i, u in enumerate(self.USER_FRAMES)
                       if u.number == s["active_uframe_num"]), 0)
            self._uframe_combo.current(ui)
            self._active_uframe = self.USER_FRAMES[ui]
            self.viewport.set_user_frame(self._active_uframe)

            # 入力欄（オーバーレイ / STONE 調整）
            for vars_, key in ((self._stl_pose_vars, "stl_vars"),
                               (self._csv_pose_vars, "csv_vars"),
                               (self._blade_pose_vars, "blade_vars"),
                               (self._stone_vars, "stone_vars")):
                for var, val in zip(vars_, s[key]):
                    var.set(val)

            self._blade_csv_path = s["blade_csv_path"]
            self._tree_programs  = s["tree_programs"]
            self._kenma_dlg_state = s["kenma_dlg"]
            self._speed_override.set(s["speed_override"])

            # ビューポートのレイヤー（STL/CSV/刃先/参照フレーム）
            self.viewport.restore_layers(s["viewport"])
            self._rf_refresh_listbox()

            # 関節角度（スライダー・FK・ロボット描画を同期）
            self._set_angles(s["joint_angles"])

            self._tree_refresh()
            self._invalidate_sim_solutions()
        finally:
            self._suppress_undo = prev

    # ──────────────────────────────────────────────────────────────────
    # 設定ファイル（エクスポート / インポート / D&D）
    # ──────────────────────────────────────────────────────────────────

    def _export_settings(self):
        """現在の設定一式を JSON ファイルへ出力する。"""
        path = filedialog.asksaveasfilename(
            title="設定をファイルへ出力",
            defaultextension=".json",
            filetypes=[("設定ファイル (JSON)", "*.json"), ("All files", "*.*")],
            initialfile="robot_sim_settings.json")
        if not path:
            return
        try:
            settings_io.save_settings(self, path)
        except Exception as e:
            messagebox.showerror("設定出力エラー",
                                 f"設定の出力に失敗しました:\n{e}")
            return
        self._set_status(
            f"✔  設定を出力: {os.path.basename(path)} — "
            "3Dビューへドラッグ＆ドロップで復元できます")

    def _import_settings(self, path: Optional[str] = None):
        """設定 JSON を読み込んで全 UI へ反映する（D&D からも呼ばれる）。"""
        if path is None:
            path = filedialog.askopenfilename(
                title="設定ファイルを読み込む",
                filetypes=[("設定ファイル (JSON)", "*.json"),
                           ("All files", "*.*")])
            if not path:
                return
        try:
            data = settings_io.load_settings(path)
        except Exception as e:
            messagebox.showerror("設定読込エラー",
                                 f"設定ファイルの読込に失敗しました:\n{e}")
            return
        with self._undo_group("設定ファイル読込"):
            try:
                warns = settings_io.apply_settings(self, data)
            except Exception as e:
                messagebox.showerror("設定反映エラー",
                                     f"設定の反映に失敗しました:\n{e}")
                return
        msg = f"✔  設定を読込: {os.path.basename(path)}"
        if warns:
            msg += f"  ⚠ {len(warns)}件の警告"
            messagebox.showwarning(
                "設定読込（警告）",
                "一部の項目を反映できませんでした:\n\n" + "\n".join(warns))
        self._set_status(msg + " — Ctrl+Z で元に戻せます")

    # ──────────────────────────────────────────────────────────────────
    # Main panels (3D viewport + route editor)
    # ──────────────────────────────────────────────────────────────────

    def _build_main_panels(self):
        # 右パネルを root に直接貼り付け（縦いっぱい）
        right_outer = ttk.Frame(self.root, width=430)
        right_outer.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 6), pady=(4, 0))
        right_outer.pack_propagate(False)

        # ── ツリーパネル（RoboDK風・左サイド） ──────────────────────────
        tree_outer = ttk.Frame(self.root, width=240)
        tree_outer.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0), pady=(4, 0))
        tree_outer.pack_propagate(False)
        self._tree_outer = tree_outer
        self._build_tree_panel(tree_outer)

        # ── ドラッグ可能な仕切り（ツリー幅をマウスで変更） ──────────────
        sash = tk.Frame(self.root, width=5, bg=BORDER,
                        cursor="sb_h_double_arrow")
        sash.pack(side=tk.LEFT, fill=tk.Y, pady=(4, 0))

        def _sash_drag(event):
            new_w = max(140, min(600, event.x_root - tree_outer.winfo_rootx()))
            tree_outer.config(width=new_w)

        def _sash_enter(event):
            sash.config(bg=ACCENT2)

        def _sash_leave(event):
            sash.config(bg=BORDER)

        sash.bind("<B1-Motion>", _sash_drag)
        sash.bind("<Enter>", _sash_enter)
        sash.bind("<Leave>", _sash_leave)
        _tip(sash, "ドラッグでステーションパネルの幅を変更できます")

        # ── 下部：折りたたみ式更新履歴（先に下端へ確保）──
        self._build_changelog_panel_collapsible(right_outer)

        # ── 上部：スクロール可能フレーム ──
        # 内容（マーカー・参照フレーム・砥石調整・経路点・STL/CSV Pose）が
        # ウィンドウ高に収まらない時でも、スクロールバー＋ホイールで
        # 末尾の STL/CSV Pose まで常に表示できるようにする（ステーション欄と同様）。
        right_canvas = tk.Canvas(right_outer, bg=BG_DARK,
                                 highlightthickness=0, bd=0)
        right_vsb = ttk.Scrollbar(right_outer, orient=tk.VERTICAL,
                                  command=right_canvas.yview)
        right_canvas.configure(yscrollcommand=right_vsb.set)
        right_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        right_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        right = ttk.Frame(right_canvas)
        right_win = right_canvas.create_window((0, 0), window=right, anchor="nw")
        self._right_canvas = right_canvas

        def _on_inner_config(event):
            right_canvas.configure(scrollregion=right_canvas.bbox("all"))

        def _on_canvas_config(event):
            # 内側フレーム幅をキャンバス幅に合わせる（横スクロール不要に）
            right_canvas.itemconfigure(right_win, width=event.width)

        right.bind("<Configure>", _on_inner_config)
        right_canvas.bind("<Configure>", _on_canvas_config)
        self._bind_right_mousewheel(right_canvas)

        # 左コンテナ：ビューポートをまとめて（ジョグパネルは右パネルへ移動）
        left_container = ttk.Frame(self.root)
        left_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0), pady=(4, 0))

        # シークバー（タイムラインスクラバー）— ビューポートの下
        self._build_seek_bar(left_container)

        # ワークフローバー（ジョグの上・ビューポートの下）
        self._build_workflow_bar(left_container)

        # 3D ビューポートは残りの全スペースを使う
        left = ttk.LabelFrame(left_container, text="  3D ビューポート — 左ドラッグ: 回転  /  右・中ドラッグ: パン  /  ホイール: カーソル位置へズーム  /  STL・CSV・設定JSON をドロップで読込")
        left.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.viewport = create_viewport(left, self.kin)

        self._build_markers_panel(right)
        self._build_ref_frames_panel(right)
        self._build_stone_adjust_panel(right)

        route_lf = ttk.LabelFrame(right,
            text="  経路点リスト (Waypoint List) — 追加・編集・削除・並べ替えが可能")
        route_lf.pack(fill=tk.X, padx=4, pady=(4, 2))
        self.route_editor = RouteEditor(
            route_lf, self.route,
            on_change=self._on_route_changed,
            on_select=self._on_waypoint_selected,
            on_before_change=self._push_undo,
            listbox_height=7,
        )
        self.route_editor.pack(fill=tk.X, padx=4, pady=4)

        self._build_overlay_panel(right)

        # 関節角度 / ジョグ操作パネル（右パネル最下部・縦積みレイアウト）
        self._build_joint_jog_panel(right, vertical=True)

        if _HAS_DND:
            self.viewport.canvas_widget.drop_target_register(DND_FILES)
            self.viewport.canvas_widget.dnd_bind("<<Drop>>", self._on_viewport_drop)

    def _bind_right_mousewheel(self, canvas):
        """ポインタが右パネル内にある間、ホイールでパネルを縦スクロールする。

        リストボックス・入力欄など自前でホイールを使う widget 上では
        そちらを優先し、二重スクロールを防ぐ（その widget で early-return）。
        """
        skip = (tk.Listbox, tk.Text, tk.Entry, ttk.Entry,
                ttk.Combobox, tk.Spinbox)

        def _wheel(event):
            w = self.root.winfo_containing(event.x_root, event.y_root)
            inside = False
            while w is not None:
                if isinstance(w, skip):
                    return        # 自前ホイール widget を優先
                if w is canvas:
                    inside = True
                    break
                w = getattr(w, "master", None)
            if not inside:
                return
            if getattr(event, "num", None) == 5 or event.delta < 0:
                canvas.yview_scroll(1, "units")
            else:
                canvas.yview_scroll(-1, "units")

        # 永続 bind_all（領域判定で右パネル外は無視）— Enter/Leave 方式の
        # 「子 widget へ移ると解除される」問題を避ける。
        self.root.bind_all("<MouseWheel>", _wheel, add="+")
        self.root.bind_all("<Button-4>",   _wheel, add="+")  # Linux 上スクロール
        self.root.bind_all("<Button-5>",   _wheel, add="+")  # Linux 下スクロール

    # ──────────────────────────────────────────────────────────────────
    # TCP・ターゲットマーカー管理パネル
    # ──────────────────────────────────────────────────────────────────

    def _build_markers_panel(self, parent):
        self._mk_list: list = []   # [{"type":"tcp"|"target","name":str,"pos":[x,y,z]}]
        self._mk_tcp_count = 0
        self._mk_tgt_count = 0

        lf = ttk.LabelFrame(parent, text="  TCP・ターゲット管理 (Markers)")
        lf.pack(fill=tk.X, padx=4, pady=(4, 2))
        _tip(lf,
             "TCP マーカーとターゲット🎯を自由に追加・削除・位置調整できます。\n"
             "「+ TCP」: 現在のロボットTCP位置にTCPマーカーを追加\n"
             "「+ 🎯」: 現在のTCP位置にターゲットを追加\n"
             "リストから選択 → X/Y/Z を編集 → 「適用」で位置を更新\n"
             "「現在TCP→」: 現在のロボットTCP座標を入力欄にセット")

        # リストボックス
        lb_frame = ttk.Frame(lf)
        lb_frame.pack(fill=tk.X, padx=6, pady=(4, 0))
        sb = tk.Scrollbar(lb_frame, orient=tk.VERTICAL)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._mk_listbox = tk.Listbox(
            lb_frame, height=4, yscrollcommand=sb.set,
            bg=BG_WIDGET, fg=FG_PRIMARY, font=("Consolas", 8),
            selectbackground=BTN_PRIMARY, selectforeground="white",
            borderwidth=0, highlightthickness=1, highlightcolor=BORDER,
            activestyle="none", exportselection=False,
        )
        self._mk_listbox.pack(fill=tk.X)
        sb.config(command=self._mk_listbox.yview)
        self._mk_listbox.bind("<<ListboxSelect>>", self._on_mk_select)
        self._mk_listbox.bind("<Double-Button-1>", self._mk_edit_selected)
        _tip(self._mk_listbox,
             "ダブルクリックでマーカー位置を編集できます\n（3Dビューへ即時反映）")

        # ボタン行（追加・削除）
        btn_row = ttk.Frame(lf)
        btn_row.pack(fill=tk.X, padx=6, pady=(3, 0))
        ttk.Button(btn_row, text="+ TCP",
                   command=self._mk_add_tcp).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="+ 🎯",
                   command=self._mk_add_target).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="削除",
                   command=self._mk_delete).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="現在TCP→", style="Primary.TButton",
                   command=self._mk_use_current_tcp).pack(side=tk.RIGHT, padx=2)

        # 位置入力行（X/Y/Z）― ホイールで ±1mm（Ctrl: ±10, Shift: ±0.1）
        pos_row = ttk.Frame(lf)
        pos_row.pack(fill=tk.X, padx=6, pady=(3, 5))
        self._mk_pos_vars = []
        for axis_i, lbl in enumerate(["X", "Y", "Z"]):
            tk.Label(pos_row, text=lbl, bg=BG_PANEL, fg=FG_SUB,
                     font=("", 8), width=2).pack(side=tk.LEFT)
            v = tk.StringVar(value="0.0")
            self._mk_pos_vars.append(v)
            ent = self._make_num_field(pos_row, v,
                                       on_change=self._mk_apply_pos_if_selected,
                                       width=7)
            ent.pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(pos_row, text="適用", style="Primary.TButton",
                   command=self._mk_apply_pos).pack(side=tk.LEFT, padx=2)

    def _mk_add_tcp(self):
        self._push_undo("TCPマーカー追加")
        self._mk_tcp_count += 1
        pos = self._mk_current_tcp_pos()
        name = f"TCP-{self._mk_tcp_count}"
        self._mk_list.append({"type": "tcp", "name": name, "pos": list(pos)})
        self._mk_refresh_listbox(select_last=True)
        self._mk_sync_viewport()
        for i, v in enumerate(self._mk_pos_vars):
            v.set(f"{pos[i]:.1f}")
        self._set_status(f"✔  TCPマーカー追加: {name}  ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})")

    def _mk_add_target(self):
        self._push_undo("ターゲット追加")
        self._mk_tgt_count += 1
        pos = self._mk_current_tcp_pos()
        name = f"Target-{self._mk_tgt_count}"
        self._mk_list.append({"type": "target", "name": name, "pos": list(pos)})
        self._mk_refresh_listbox(select_last=True)
        self._mk_sync_viewport()
        for i, v in enumerate(self._mk_pos_vars):
            v.set(f"{pos[i]:.1f}")
        self._set_status(f"✔  ターゲット追加: {name}  ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})")

    def _mk_delete(self):
        sel = self._mk_listbox.curselection()
        if not sel:
            self._set_status("⚠  削除するマーカーをリストから選択してください")
            return
        idx = sel[0]
        name = self._mk_list[idx]["name"]
        self._push_undo("マーカー削除")
        del self._mk_list[idx]
        self._mk_refresh_listbox()
        self._mk_sync_viewport()
        self._set_status(f"✔  マーカー削除: {name}")

    def _mk_apply_pos(self):
        sel = self._mk_listbox.curselection()
        if not sel:
            self._set_status("⚠  リストからマーカーを選択してください")
            return
        try:
            pos = [float(v.get()) for v in self._mk_pos_vars]
        except ValueError:
            self._set_status("⚠  数値を入力してください")
            return
        idx = sel[0]
        self._push_undo("マーカー位置変更", coalesce=True)
        self._mk_list[idx]["pos"] = pos
        self._mk_refresh_listbox(select_idx=idx)
        self._mk_sync_viewport()
        name = self._mk_list[idx]["name"]
        self._set_status(f"✔  マーカー位置更新: {name}  ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})")

    def _mk_use_current_tcp(self):
        pos = self._mk_current_tcp_pos()
        for i, v in enumerate(self._mk_pos_vars):
            v.set(f"{pos[i]:.1f}")
        sel = self._mk_listbox.curselection()
        if sel:
            self._mk_apply_pos()
        else:
            self._set_status(f"✔  現在TCP位置: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

    def _mk_apply_pos_if_selected(self):
        """選択中のマーカーがあれば位置を即時反映する（ホイール/Enter用）。"""
        if self._mk_listbox.curselection():
            self._mk_apply_pos()

    def _mk_edit_selected(self, event=None):
        """マーカーをダブルクリックで位置編集（X/Y/Z、ライブ反映）。"""
        sel = self._mk_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx >= len(self._mk_list):
            return
        m = self._mk_list[idx]
        from types import SimpleNamespace
        obj = SimpleNamespace(
            x=round(m["pos"][0]),
            y=round(m["pos"][1]),
            z=round(m["pos"][2]),
        )

        def _apply():
            m["pos"] = [obj.x, obj.y, obj.z]
            self._mk_refresh_listbox(select_idx=idx)
            self._mk_sync_viewport()
            for i, v in enumerate(self._mk_pos_vars):
                v.set(f"{m['pos'][i]:.1f}")

        self._frame_editor_dialog(
            title=f"マーカー編集: {m['name']}",
            desc="マーカーの位置を編集します（3Dビューへ即時反映）。\nホイール: ±1mm  /  Shift+ホイール: ±0.1mm  /  Ctrl+ホイール: ±10mm",
            obj=obj,
            fields=["x", "y", "z"],
            labels=["X 位置 (mm)", "Y 位置 (mm)", "Z 位置 (mm)"],
            on_apply=_apply,
            undo_label="マーカー編集",
            fmt="{:.0f}")

    def _on_mk_select(self, event=None):
        if getattr(self, '_mk_refreshing', False):
            return
        sel = self._mk_listbox.curselection()
        if not sel:
            return
        pos = self._mk_list[sel[0]]["pos"]
        for i, v in enumerate(self._mk_pos_vars):
            v.set(f"{pos[i]:.1f}")

    def _mk_current_tcp_pos(self) -> np.ndarray:
        T = self.kin.forward(self._joint_angles)
        if self._active_tool and self._active_tool.z != 0.0:
            T = T @ self._active_tool.to_transform()
        return T[:3, 3]

    def _mk_refresh_listbox(self, select_last: bool = False, select_idx: Optional[int] = None):
        self._mk_refreshing = True
        try:
            self._mk_listbox.delete(0, tk.END)
            for m in self._mk_list:
                p = m["pos"]
                prefix = "TCP" if m["type"] == "tcp" else "🎯 "
                entry = f"{prefix}  {m['name']:10s}  ({p[0]:7.1f}, {p[1]:7.1f}, {p[2]:7.1f})"
                self._mk_listbox.insert(tk.END, entry)
                color = "#00FFCC" if m["type"] == "tcp" else "#FF8800"
                self._mk_listbox.itemconfig(tk.END, fg=color)
            if select_last and self._mk_listbox.size() > 0:
                self._mk_listbox.selection_set(tk.END)
                self._mk_listbox.see(tk.END)
            elif select_idx is not None and 0 <= select_idx < self._mk_listbox.size():
                self._mk_listbox.selection_set(select_idx)
        finally:
            self._mk_refreshing = False

    def _mk_sync_viewport(self):
        tcp_markers = [
            {"name": m["name"], "pos": m["pos"]}
            for m in self._mk_list if m["type"] == "tcp"
        ]
        target_markers = [
            {"name": m["name"], "pos": m["pos"]}
            for m in self._mk_list if m["type"] == "target"
        ]
        self.viewport.set_markers(tcp_markers, target_markers)

    # ──────────────────────────────────────────────────────────────────
    # 参照フレームパネル (Reference Frames)
    # ──────────────────────────────────────────────────────────────────

    def _build_ref_frames_panel(self, parent):
        """参照フレーム（座標軸）の管理パネル。"""
        lf = ttk.LabelFrame(parent, text="  参照フレーム (Reference Frames)")
        lf.pack(fill=tk.X, padx=4, pady=(4, 2))
        _tip(lf, "名前付き座標フレームを3Dビューポートに表示します。\n"
                 "各フレームはX(赤)/Y(緑)/Z(青)軸と菱形マーカーで表示されます。")

        lb_frame = ttk.Frame(lf)
        lb_frame.pack(fill=tk.X, padx=6, pady=(4, 0))
        sb = tk.Scrollbar(lb_frame, orient=tk.VERTICAL)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._rf_listbox = tk.Listbox(
            lb_frame, height=3, yscrollcommand=sb.set,
            bg=BG_WIDGET, fg=FG_PRIMARY, font=("Consolas", 8),
            selectbackground=BTN_PRIMARY, selectforeground="white",
            borderwidth=0, highlightthickness=1, highlightcolor=BORDER,
            activestyle="none",
        )
        self._rf_listbox.pack(fill=tk.X)
        sb.config(command=self._rf_listbox.yview)
        self._rf_listbox.bind("<Double-Button-1>", self._rf_edit_selected)
        _tip(self._rf_listbox,
             "ダブルクリックでフレームの位置・姿勢を編集できます\n"
             "（3Dビューへ即時反映）")

        btn_row = ttk.Frame(lf)
        btn_row.pack(fill=tk.X, padx=6, pady=(3, 5))
        ttk.Button(btn_row, text="+ フレーム追加",
                   command=self._add_ref_frame_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="削除",
                   command=self._rf_delete).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="全クリア",
                   command=self._rf_clear_all).pack(side=tk.LEFT, padx=2)

    def _rf_refresh_listbox(self):
        self._rf_listbox.delete(0, tk.END)
        for rf in self.viewport.get_ref_frames():
            T = rf["T"]
            ox, oy, oz = T[0, 3], T[1, 3], T[2, 3]
            entry = f"{rf['name']:12s}  ({ox:.0f}, {oy:.0f}, {oz:.0f})"
            self._rf_listbox.insert(tk.END, entry)
            self._rf_listbox.itemconfig(tk.END, fg=rf.get("color", "#FF88FF"))

    def _rf_delete(self):
        sel = self._rf_listbox.curselection()
        if not sel:
            self._set_status("⚠  削除するフレームをリストから選択してください")
            return
        idx = sel[0]
        frames = self.viewport.get_ref_frames()
        if idx < len(frames):
            name = frames[idx]["name"]
            self._push_undo("参照フレーム削除")
            self.viewport.remove_ref_frame(name)
            self._rf_refresh_listbox()
            self._set_status(f"✔  参照フレーム削除: {name}")

    def _rf_clear_all(self):
        self._push_undo("参照フレーム全クリア")
        self.viewport.clear_ref_frames()
        self._rf_refresh_listbox()
        self._set_status("✔  参照フレームをすべてクリア")

    def _rf_edit_selected(self, event=None):
        """参照フレームをダブルクリックで編集（X/Y/Z/Rx/Ry/Rz、ライブ反映）。"""
        sel = self._rf_listbox.curselection()
        if not sel:
            return
        frames = self.viewport.get_ref_frames()
        idx = sel[0]
        if idx >= len(frames):
            return
        rf = frames[idx]
        name  = rf["name"]
        color = rf.get("color", "#FF88FF")
        from types import SimpleNamespace
        x, y, z, rx, ry, rz = Kinematics.transform_to_pose(rf["T"])
        obj = SimpleNamespace(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz)

        def _apply():
            if name == "UF9: STONE":
                # UF9 は専用の同期ヘルパー経由（USER_FRAMES 等も更新）
                uf9 = self._uf9_frame(x=obj.x, y=obj.y, z=obj.z,
                                      rx=obj.rx, ry=obj.ry, rz=obj.rz)
                self._sync_uf9(uf9)
            else:
                self.viewport.remove_ref_frame(name)
                self.viewport.add_ref_frame(name, obj.x, obj.y, obj.z,
                                            obj.rx, obj.ry, obj.rz,
                                            color=color)
                self._rf_refresh_listbox()

        self._frame_editor_dialog(
            title=f"参照フレーム編集: {name}",
            desc="参照フレームの位置・姿勢を編集します（3Dビューへ即時反映）。",
            obj=obj,
            fields=["x", "y", "z", "rx", "ry", "rz"],
            labels=["X 位置 (mm)", "Y 位置 (mm)", "Z 位置 (mm)",
                    "Rx 回転 (°)", "Ry 回転 (°)", "Rz 回転 (°)"],
            on_apply=_apply,
            undo_label="参照フレーム編集")

    def _add_ref_frame_dialog(self):
        """Show dialog to add a custom reference frame."""
        win = tk.Toplevel(self.root)
        win.title("参照フレームを追加")
        win.geometry("380x280")
        win.configure(bg=BG_DARK)
        win.resizable(False, False)

        tk.Label(win, text="参照フレームを追加",
                 bg=BG_DARK, fg=ACCENT,
                 font=("Yu Gothic UI", 11, "bold")).pack(pady=(12, 4), padx=16, anchor="w")

        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=16)

        fields = {}
        labels = ["名前", "X (mm)", "Y (mm)", "Z (mm)", "Rx (deg)", "Ry (deg)", "Rz (deg)"]
        defaults = ["MyFrame", "0", "0", "0", "0", "0", "0"]
        for lbl, dflt in zip(labels, defaults):
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=lbl, bg=BG_PANEL, fg=FG_SUB,
                     font=("Yu Gothic UI", 9), width=10, anchor="w").pack(side=tk.LEFT)
            v = tk.StringVar(value=dflt)
            fields[lbl] = v
            if lbl == "名前":
                ttk.Entry(row, textvariable=v, width=18).pack(side=tk.LEFT, padx=4)
            else:
                self._make_num_field(row, v, width=18).pack(side=tk.LEFT, padx=4)

        def _apply():
            name = fields["名前"].get().strip() or "Frame"
            try:
                x  = float(fields["X (mm)"].get())
                y  = float(fields["Y (mm)"].get())
                z  = float(fields["Z (mm)"].get())
                rx = float(fields["Rx (deg)"].get())
                ry = float(fields["Ry (deg)"].get())
                rz = float(fields["Rz (deg)"].get())
            except ValueError:
                messagebox.showerror("入力エラー", "数値を入力してください", parent=win)
                return
            self._push_undo("参照フレーム追加")
            self.viewport.add_ref_frame(name, x, y, z, rx, ry, rz)
            self._rf_refresh_listbox()
            self._set_status(f"✔  参照フレーム追加: {name}  ({x:.0f}, {y:.0f}, {z:.0f})")
            win.destroy()

        btn_row = ttk.Frame(win)
        btn_row.pack(pady=8)
        ttk.Button(btn_row, text="追加", style="Primary.TButton",
                   command=_apply).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_row, text="キャンセル",
                   command=win.destroy).pack(side=tk.LEFT, padx=6)
        win.bind("<Return>", lambda e: _apply())

    # ──────────────────────────────────────────────────────────────────
    # UF9 STONE 位置調整パネル（即時反映）
    # ──────────────────────────────────────────────────────────────────

    def _build_stone_adjust_panel(self, parent):
        """UF9 STONE 参照フレームを X/Y/Z/Rx/Ry/Rz で即時調整するパネル。"""
        lf = ttk.LabelFrame(parent, text="  🪨 UF9 STONE 位置調整 (即時反映)")
        lf.pack(fill=tk.X, padx=4, pady=(2, 2))
        _tip(lf,
             "砥石 (UF9 STONE) 参照フレームの位置・姿勢を調整します。\n"
             "X/Y/Z: 位置(mm)  Rx/Ry/Rz: 姿勢(°)\n"
             "入力欄でマウスホイール → 3Dビューへ即時反映\n"
             "Ctrl+ホイール=10, Shift+ホイール=0.1, 通常=1\n"
             "ここで設定した値は kenma 生成の砥石接触座標として使われます。")

        inner = ttk.Frame(lf)
        inner.pack(fill=tk.X, padx=4, pady=2)
        self._stone_vars: list = []
        # 現在の UF9 から初期値を取得（なければ既定値）
        uf9 = self._get_uf9()
        defaults = [uf9.x, uf9.y, uf9.z, uf9.rx, uf9.ry, uf9.rz]
        for i, axis in enumerate(["X", "Y", "Z", "Rx", "Ry", "Rz"]):
            r, c = divmod(i, 3)
            tk.Label(inner, text=axis, bg=BG_PANEL, fg=FG_SUB,
                     font=("", 8), width=3).grid(row=r, column=c*2, padx=1)
            v = tk.StringVar(value=f"{defaults[i]:.1f}")
            self._stone_vars.append(v)
            ent = self._make_num_field(inner, v,
                                       on_change=self._apply_stone_adjust, width=6)
            ent.grid(row=r, column=c*2+1, padx=1, pady=1)

        btn_row = ttk.Frame(lf)
        btn_row.pack(padx=4, pady=(0, 3))
        ttk.Button(btn_row, text="適用", style="Primary.TButton",
                   command=self._apply_stone_adjust).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="現在UF9→",
                   command=self._refresh_stone_fields).pack(side=tk.LEFT, padx=2)

    def _get_uf9(self):
        """現在の UF9 STONE を返す（USER_FRAMES → 既定値）。存在しなければ既定値の UF を返す（追加はしない）。"""
        for uf in self.USER_FRAMES:
            if uf.number == 9:
                return uf
        return UserFrame.stone9()

    @staticmethod
    def _uf9_frame(**overrides) -> UserFrame:
        """UF9 STONE を生成する（既定値の単一ソース = UserFrame.stone9）。

        number/name/comment と未指定の座標は stone9() の既定値を使う。
        """
        from dataclasses import replace
        return replace(UserFrame.stone9(), **overrides)

    def _sync_uf9(self, uf9, status: Optional[str] = None,
                  update_fields: bool = True):
        """UF9 STONE を全箇所へ同期する（単一の同期ヘルパー）。

        USER_FRAMES・UFrame コンボボックス・アクティブ UFrame・
        ビューポート参照フレーム "UF9: STONE"・STONE 調整パネル入力欄・
        ステーションツリー を一括更新する。kenma 生成（_kenma_inputs）が
        読み取る座標と常に一致させる。
        """
        nums = [uf.number for uf in self.USER_FRAMES]
        if 9 in nums:
            self.USER_FRAMES[nums.index(9)] = uf9
        else:
            self.USER_FRAMES.append(uf9)

        # コンボボックスの選択肢を更新
        if hasattr(self, "_uframe_combo"):
            uf_names = [f"UF{u.number}: {u.name}" for u in self.USER_FRAMES]
            self._uframe_combo.config(values=uf_names)

        # アクティブ UFrame が UF9 なら差し替えてビューを更新
        if self._active_uframe is not None and self._active_uframe.number == 9:
            self._active_uframe = uf9
            self.viewport.set_user_frame(uf9)

        # ビューポートの参照フレーム "UF9: STONE" を再登録（即時再描画）
        self.viewport.remove_ref_frame("UF9: STONE")
        self.viewport.add_ref_frame("UF9: STONE",
                                    uf9.x, uf9.y, uf9.z,
                                    uf9.rx, uf9.ry, uf9.rz,
                                    color="#C586C0")
        if hasattr(self, "_rf_listbox"):
            self._rf_refresh_listbox()

        # STONE 位置調整パネルの入力欄を同期
        if (update_fields and hasattr(self, "_stone_vars")
                and len(self._stone_vars) >= 6):
            for var, val in zip(self._stone_vars,
                                [uf9.x, uf9.y, uf9.z, uf9.rx, uf9.ry, uf9.rz]):
                var.set(f"{val:.1f}")

        if hasattr(self, "_tree"):
            self._tree_refresh()
        if status:
            self._set_status(status)

    def _refresh_stone_fields(self):
        """入力欄を現在の UF9 STONE 値で再設定する。"""
        uf9 = self._get_uf9()
        vals = [uf9.x, uf9.y, uf9.z, uf9.rx, uf9.ry, uf9.rz]
        for v, val in zip(self._stone_vars, vals):
            v.set(f"{val:.1f}")
        self._set_status("✔  UF9 STONE の現在値を入力欄に反映しました")

    def _apply_stone_adjust(self):
        """入力欄の値で UF9 STONE を更新し、3Dビューへ即時反映する。

        USER_FRAMES の UF9 と、ビューポートの参照フレーム / アクティブ UFrame を
        すべて同期させる（kenma 生成が読み取る座標と一致させる）。
        """
        try:
            x, y, z, rx, ry, rz = [float(v.get()) for v in self._stone_vars]
        except (ValueError, tk.TclError):
            self._set_status("⚠  数値を入力してください")
            return

        self._push_undo("UF9 STONE 調整", coalesce=True)
        uf9 = self._uf9_frame(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz)
        self._sync_uf9(
            uf9, update_fields=False,
            status=f"✔  UF9 STONE 更新: X={x:.1f} Y={y:.1f} Z={z:.1f} Rz={rz:.1f}°")

    # ──────────────────────────────────────────────────────────────────
    # 更新履歴パネル（右サイドバー下部）
    # ──────────────────────────────────────────────────────────────────

    def _build_changelog_panel_collapsible(self, parent):
        """折りたたみ式更新履歴パネル（右パネル下部固定）。"""
        container = ttk.Frame(parent)
        container.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=(0, 4))

        # ▶/▼ トグルヘッダー
        self._cl_expanded = tk.BooleanVar(value=False)
        header = tk.Frame(container, bg=BG_WIDGET, cursor="hand2")
        header.pack(fill=tk.X)
        self._cl_toggle_lbl = tk.Label(
            header,
            text=f"▶  更新履歴 — 最新: v{APP_VERSION}",
            bg=BG_WIDGET, fg=ACCENT,
            font=("", 9, "bold"), anchor="w", padx=6, pady=4,
        )
        self._cl_toggle_lbl.pack(fill=tk.X)

        # 本文フレーム（初期非表示）
        body = tk.Frame(container, bg="#111111")
        self._cl_body = body

        txt_frame = ttk.Frame(body)
        txt_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        sb = tk.Scrollbar(txt_frame, orient=tk.VERTICAL)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        txt = tk.Text(
            txt_frame, height=10,
            bg="#111111", fg="#CCCCCC",
            font=("Consolas", 9),
            wrap=tk.WORD, borderwidth=0,
            highlightthickness=0, state="normal",
            cursor="arrow", selectbackground=BTN_PRIMARY,
            yscrollcommand=sb.set,
        )
        txt.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        sb.config(command=txt.yview)

        for ver, date, time_, items in CHANGELOG:
            txt.insert(tk.END, f"v{ver}  {date}  {time_}\n", "ver")
            for item in items:
                txt.insert(tk.END, f"  • {item}\n", "item")
            txt.insert(tk.END, "\n")
        txt.tag_config("ver",  foreground="#F5C400", font=("Consolas", 9, "bold"))
        txt.tag_config("item", foreground="#CCCCCC")
        txt.config(state="disabled")

        def _toggle(event=None):
            if self._cl_expanded.get():
                self._cl_body.pack_forget()
                self._cl_expanded.set(False)
                self._cl_toggle_lbl.config(text=f"▶  更新履歴 — 最新: v{APP_VERSION}")
            else:
                self._cl_body.pack(fill=tk.BOTH, expand=True)
                self._cl_expanded.set(True)
                self._cl_toggle_lbl.config(text=f"▼  更新履歴 — 最新: v{APP_VERSION}")

        header.bind("<Button-1>", _toggle)
        self._cl_toggle_lbl.bind("<Button-1>", _toggle)

    # ──────────────────────────────────────────────────────────────────
    # 関節角度スライダー + 速度オーバーライド + UTool / UFrame
    # ──────────────────────────────────────────────────────────────────

    def _build_joint_jog_panel(self, parent=None, vertical=False):
        """関節角度スライダーとジョグ操作を1パネルに統合（折りたたみ可）。

        vertical=True のときは右パネル（幅~430px）向けに3列を縦積みし、
        スライダー長も短くして横幅に収める。
        """
        if parent is None:
            parent = self.root

        # 縦積み時の各列の pack 引数（右パネルに収める）
        self._jog_vertical = vertical
        col_pack = (dict(side=tk.TOP, fill=tk.X, pady=(4, 0))
                    if vertical else dict(side=tk.LEFT, fill=tk.Y))
        slider_len = 150 if vertical else 240

        # 折りたたみコンテナ（更新履歴と同様の構造）
        container = ttk.Frame(parent)
        if vertical:
            container.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(4, 2))
        else:
            container.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=(2, 0))

        self._jog_expanded = tk.BooleanVar(value=True)
        jog_hdr = tk.Frame(container, bg=BG_WIDGET, cursor="hand2")
        jog_hdr.pack(fill=tk.X)
        self._jog_toggle_lbl = tk.Label(
            jog_hdr, text="▼  関節角度 / ジョグ操作",
            bg=BG_WIDGET, fg=ACCENT2,
            font=("", 9, "bold"), anchor="w", padx=6, pady=3)
        self._jog_toggle_lbl.pack(fill=tk.X)

        outer = ttk.Frame(container)
        outer.pack(fill=tk.X)  # 初期状態: 展開

        def _toggle_jog(event=None):
            if self._jog_expanded.get():
                outer.pack_forget()
                self._jog_expanded.set(False)
                self._jog_toggle_lbl.config(text="▶  関節角度 / ジョグ操作")
            else:
                outer.pack(fill=tk.X)
                self._jog_expanded.set(True)
                self._jog_toggle_lbl.config(text="▼  関節角度 / ジョグ操作")

        jog_hdr.bind("<Button-1>", _toggle_jog)
        self._jog_toggle_lbl.bind("<Button-1>", _toggle_jog)

        # ---- 関節スライダー + ジョグボタン（統合パネル） ----
        slider_lf = ttk.LabelFrame(outer, text="  関節角度 / ジョグ操作")
        slider_lf.pack(**col_pack)

        # モード選択 + ステップ幅（ヘッダー行）
        header = ttk.Frame(slider_lf)
        header.pack(fill=tk.X, padx=6, pady=(3, 1))

        self._jog_mode = tk.StringVar(value="Joint")
        ttk.Radiobutton(header, text="● Joint（関節）",
                        variable=self._jog_mode, value="Joint").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(header, text="○ Cartesian（直交）",
                        variable=self._jog_mode, value="Cartesian").pack(side=tk.LEFT, padx=4)

        ttk.Separator(header, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        tk.Label(header, text="ステップ幅:", bg=BG_PANEL, fg=FG_SUB, font=("", 8)).pack(side=tk.LEFT)
        self._jog_step = tk.StringVar(value="5")
        ttk.Combobox(header, textvariable=self._jog_step,
                     values=["0.5", "1", "5", "10", "45"],
                     width=5, state="readonly").pack(side=tk.LEFT, padx=2)
        tk.Label(header, text="° / mm", bg=BG_PANEL, fg=FG_SUB, font=("", 8)).pack(side=tk.LEFT)

        lower, upper = self.kin.dh.get_joint_limits_deg()
        speeds = self.kin.dh.get_joint_max_speeds()
        self._slider_vars = []
        self._jog_axis_labels = []

        JOINT_TIPS = [
            "J1 — 旋回軸\n胴体全体が水平に回転します\n可動範囲: ±170°  最大速度: 210°/s",
            "J2 — 肩軸\n上腕が前後方向に動きます\n可動範囲: -85°〜+145°  最大速度: 210°/s",
            "J3 — 肘軸\n前腕が上下に動きます\n可動範囲: -175°〜+255°  最大速度: 275°/s",
            "J4 — 前腕回転軸\n前腕がねじれ回転します\n可動範囲: ±190°  最大速度: 400°/s",
            "J5 — 手首ピッチ軸\n手首が上下に傾きます\n可動範囲: ±135°  最大速度: 400°/s",
            "J6 — 手首回転軸\nツール（包丁）が軸回転します\n可動範囲: ±360°  最大速度: 600°/s",
        ]

        for i in range(6):
            row = ttk.Frame(slider_lf)
            row.pack(fill=tk.X, padx=6, pady=1)

            init_deg = np.rad2deg(self._joint_angles[i])
            var = tk.DoubleVar(value=init_deg)
            self._slider_vars.append(var)

            # 軸名ラベル（Joint/Cartesianモードで切替）
            jlbl = tk.Label(row, text=f"J{i+1}", width=3,
                            bg=BG_PANEL, fg=ACCENT2,
                            font=("Consolas", 9, "bold"), anchor="center")
            jlbl.pack(side=tk.LEFT, padx=(0, 2))
            _tip(jlbl, JOINT_TIPS[i])
            self._jog_axis_labels.append((jlbl, f"J{i+1}", ["X","Y","Z","Rx","Ry","Rz"][i]))

            # 水平スライダー
            sc = ttk.Scale(row, from_=lower[i], to=upper[i],
                           variable=var, orient=tk.HORIZONTAL, length=slider_len,
                           command=lambda val, idx=i: self._on_slider_change(idx, float(val)))
            sc.pack(side=tk.LEFT)
            _tip(sc, JOINT_TIPS[i])

            # 現在角度表示（StringVar で科学記数法バグを回避）
            angle_str_var = tk.StringVar(value=f"{init_deg:7.1f}")
            def _make_trace(dv, sv):
                def _trace(*_):
                    sv.set(f"{dv.get():7.1f}")
                return _trace
            var.trace_add("write", _make_trace(var, angle_str_var))
            tk.Label(row, textvariable=angle_str_var,
                     bg=BG_PANEL, fg=FG_PRIMARY,
                     font=("Consolas", 8), width=8, anchor="e").pack(side=tk.LEFT, padx=2)
            tk.Label(row, text="°",
                     bg=BG_PANEL, fg=FG_SUB, font=("", 8)).pack(side=tk.LEFT)

            # ◀▶ ジョグボタン
            ttk.Button(row, text="◀", style="Jog.TButton",
                       command=lambda ax=i: self._jog(ax, -1)).pack(side=tk.LEFT, padx=(4, 1))
            ttk.Button(row, text="▶", style="Jog.TButton",
                       command=lambda ax=i: self._jog(ax, +1)).pack(side=tk.LEFT, padx=(1, 4))

            # 可動範囲・速度（薄いメタ情報）
            tk.Label(row,
                text=f"  {lower[i]:.0f}〜{upper[i]:.0f}°   {speeds[i]:.0f}°/s",
                bg=BG_PANEL, fg=FG_SUB, font=("", 7), anchor="w"
            ).pack(side=tk.LEFT, padx=(4, 0))

        # モード切替でラベルを更新
        def _update_jog_labels(*_):
            mode = self._jog_mode.get()
            for lbl, jname, cname in self._jog_axis_labels:
                lbl.config(text=jname if mode == "Joint" else cname)
        self._jog_mode.trace_add("write", _update_jog_labels)

        # ---- 中間列：ファイル I/O + シミュレーション + IK ----
        mid_col = ttk.Frame(outer)
        mid_col.pack(**col_pack)

        # ファイル I/O
        io_lf = ttk.LabelFrame(mid_col, text="  ファイル (File I/O)")
        io_lf.pack(fill=tk.X, pady=(4, 2))
        _tip(io_lf,
             "経路データの保存・読込とFANUCコントローラへの出力\n\n"
             "📂 CSV 読込: 保存済みの経路ファイルを開く  (Ctrl+O)\n"
             "💾 CSV 保存: 現在の経路をCSVファイルに保存  (Ctrl+S)\n"
             "📤 TP 出力: FANUC TP プログラム(.ls)を生成  (Ctrl+E)\n"
             "  → コントローラへFTP/USBで転送して実機動作可能")
        io_inner = ttk.Frame(io_lf)
        io_inner.pack(padx=6, pady=4)
        btn_csv_load = ttk.Button(io_inner, text="📂 CSV 読込", command=self._load_csv)
        btn_csv_load.pack(pady=1, fill=tk.X)
        _tip(btn_csv_load, "保存済みの経路CSVファイルを開きます  (Ctrl+O)")
        btn_csv_save = ttk.Button(io_inner, text="💾 CSV 保存", command=self._save_csv)
        btn_csv_save.pack(pady=1, fill=tk.X)
        _tip(btn_csv_save, "現在の経路をCSVファイルに保存します  (Ctrl+S)")
        btn_tp = ttk.Button(io_inner, text="📤 TP 出力", command=self._export_tp)
        btn_tp.pack(pady=1, fill=tk.X)
        _tip(btn_tp,
             "FANUC TP プログラム (.ls) を生成します  (Ctrl+E)\n"
             "IK 計算 → 全経路点の関節角度を自動算出\n"
             "生成ファイルをコントローラへ転送することで実機動作が可能です")

        # シミュレーション
        sim_lf = ttk.LabelFrame(mid_col, text="  シミュレーション")
        sim_lf.pack(fill=tk.X, pady=2)
        _tip(sim_lf,
             "設定した経路点を順番にIK計算しながらアニメーション表示します。\n\n"
             "▶ 実行: 経路の先頭から順にロボットを動かす  (F5)\n"
             "■ 停止: 途中で停止する\n\n"
             "速度オーバーライドの値がアニメーション速度に反映されます。\n"
             "IK 失敗した点はスキップされます（ステータスバーに表示）")
        sim_inner = ttk.Frame(sim_lf)
        sim_inner.pack(padx=6, pady=4)
        self._sim_btn = ttk.Button(sim_inner, text="▶  実行 (F5)",
                                   style="Primary.TButton",
                                   command=self._start_simulation)
        self._sim_btn.pack(pady=1, fill=tk.X)
        ttk.Button(sim_inner, text="■  停止",
                   style="Danger.TButton",
                   command=self._stop_simulation).pack(pady=1, fill=tk.X)
        self._smooth_btn = ttk.Button(
            sim_inner, text="🎬  滑らか再生（事前描画）",
            command=self._smooth_play)
        self._smooth_btn.pack(pady=1, fill=tk.X)
        _tip(self._smooth_btn,
             "再生前に全フレームを画像として事前描画し、再生中は重い 3D\n"
             "再描画をやめて画像を差し替えるだけにします（最も滑らか・高品質）。\n"
             "・初回のみ「描画中…」の待ち時間が発生します（キャンセル可）\n"
             "・2回目以降はキャッシュから即再生（ルート変更時は再描画）\n"
             "・実機メッシュのまま滑らかに再生できます（軽量表示への切替不要）")
        fps_row = ttk.Frame(sim_inner)
        fps_row.pack(fill=tk.X, pady=(1, 0))
        tk.Label(fps_row, text="FPS:", bg=BG_PANEL, fg=FG_SUB,
                 font=("", 8)).pack(side=tk.LEFT)
        self._pre_fps_var = tk.IntVar(value=int(self._pre_fps))
        fps_spin = tk.Spinbox(
            fps_row, from_=5, to=30, increment=1,
            textvariable=self._pre_fps_var, width=4, font=("", 8),
            bg=BG_WIDGET, fg=FG_PRIMARY, insertbackground=FG_PRIMARY,
            command=self._on_pre_fps_change)
        fps_spin.pack(side=tk.LEFT, padx=2)
        fps_spin.bind("<Return>", lambda e: self._on_pre_fps_change())
        _tip(fps_spin,
             "事前描画のフレームレートを設定します（5〜30 fps）。\n"
             "低くするほど事前描画が速く終わります（画質は下がります）。\n"
             "変更するとフレームキャッシュが無効化され再描画が必要になります。")
        tk.Label(fps_row, text="fps（事前描画）", bg=BG_PANEL, fg=FG_SUB,
                 font=("", 8)).pack(side=tk.LEFT, padx=(0, 2))
        self._save_video_btn = ttk.Button(
            sim_inner, text="💾  動画保存（MP4 / GIF）",
            command=self._save_smooth_video, state="disabled")
        self._save_video_btn.pack(pady=1, fill=tk.X)
        _tip(self._save_video_btn,
             "事前描画したフレームを動画ファイルとして保存します。\n"
             "・MP4: 高品質（imageio-ffmpeg が使用可能な場合）\n"
             "・GIF: 256色、フォールバック\n"
             "「滑らか再生」を一度実行するとボタンが有効になります。")
        self._sim_progress_var = tk.StringVar(value="待機中")
        tk.Label(sim_inner, textvariable=self._sim_progress_var,
                 bg=BG_PANEL, fg=FG_SUB, font=("", 7)).pack()
        # 再生時間（経過時間／推定総時間）— フリーズ検知用に常時更新
        self._sim_time_var = tk.StringVar(value="⏱  0.0s / 0.0s")
        tk.Label(sim_inner, textvariable=self._sim_time_var,
                 bg=BG_PANEL, fg=ACCENT, font=("", 9, "bold")).pack()
        # 軽量表示トグル（停止中・再生中いずれも即時切替）
        self._fast_mode_var = tk.BooleanVar(value=False)
        fm_cb = tk.Checkbutton(
            sim_inner, text="軽量表示（高速・円柱）",
            variable=self._fast_mode_var,
            command=self._toggle_fast_mode,
            bg=BG_PANEL, fg=FG_SUB, activebackground=BG_PANEL,
            selectcolor=BG_WIDGET, font=("", 8))
        fm_cb.pack(anchor="w", pady=(2, 0))
        _tip(fm_cb,
             "ON: ロボットを円柱ジオメトリで高速描画（描画が重いPC向け）。\n"
             "OFF: 実機メッシュで描画（高品質）。\n"
             "停止中・再生中いずれもチェックで即座に切り替わります。")
        # 案2: 再生中は自動で軽量表示へ切り替え（停止時に元へ戻す）
        self._auto_fast_var = tk.BooleanVar(value=True)
        af_cb = tk.Checkbutton(
            sim_inner, text="再生中は自動で軽量表示",
            variable=self._auto_fast_var,
            bg=BG_PANEL, fg=FG_SUB, activebackground=BG_PANEL,
            selectcolor=BG_WIDGET, font=("", 8))
        af_cb.pack(anchor="w", pady=(0, 0))
        _tip(af_cb,
             "ON: シミュレーション再生の間だけ自動で軽量表示（円柱）に切り替え、\n"
             "    停止すると実機メッシュ表示へ戻します。滑らかな再生向け。\n"
             "OFF: 再生中も上の「軽量表示」設定のまま描画します。")

        # 逆運動学 (IK)
        ik_lf = ttk.LabelFrame(mid_col, text="  逆運動学 (IK)")
        ik_lf.pack(fill=tk.X, pady=(2, 4))
        _tip(ik_lf,
             "逆運動学 (IK: Inverse Kinematics)\n"
             "経路点の位置・姿勢から関節角度を自動計算してロボットを移動させます。\n\n"
             "使い方:\n"
             "1. 経路点番号を入力（右のリストと対応）\n"
             "2. 「IK 計算 → 移動」をクリック\n"
             "3. 関節スライダーと3Dビューが更新されます\n\n"
             "IK 失敗: 指定位置がロボットの可動範囲外の場合に発生します")
        ik_inner = ttk.Frame(ik_lf)
        ik_inner.pack(padx=6, pady=4)
        wp_row = ttk.Frame(ik_inner)
        wp_row.pack(fill=tk.X, pady=2)
        tk.Label(wp_row, text="経路点 P[",
                 bg=BG_PANEL, fg=FG_SUB, font=("", 8)).pack(side=tk.LEFT)
        self._ik_wp_var = tk.IntVar(value=1)
        ttk.Spinbox(wp_row, from_=1, to=999,
                    textvariable=self._ik_wp_var, width=4).pack(side=tk.LEFT)
        tk.Label(wp_row, text="]",
                 bg=BG_PANEL, fg=FG_SUB, font=("", 8)).pack(side=tk.LEFT)
        ik_btn = ttk.Button(ik_inner, text="IK 計算 → 移動",
                            command=self._compute_ik_for_wp)
        ik_btn.pack(pady=2, fill=tk.X)
        _tip(ik_btn, "指定した経路点にロボットを移動させます\n解析解+数値解フォールバックでIKを解きます")

        # ---- 右列：速度OVR + UTool + UFrame ----
        right_col = ttk.Frame(outer)
        right_col.pack(**col_pack)

        # 速度オーバーライド
        spd_lf = ttk.LabelFrame(right_col, text="  速度オーバーライド")
        spd_lf.pack(fill=tk.X, pady=(4, 2))
        _tip(spd_lf,
             "全軸の速度上限をパーセントで設定します。\n"
             "100% = 最大速度（J1: 210°/s, J6: 600°/s など）\n"
             "シミュレーションと TP 出力の両方に反映されます。\n"
             "初回確認時は低い値から始めることを推奨します。")

        spd_inner = ttk.Frame(spd_lf)
        spd_inner.pack(fill=tk.X, padx=6, pady=4)
        self._speed_override = tk.IntVar(value=100)
        ovr_sc = ttk.Scale(spd_inner, from_=1, to=100,
                           variable=self._speed_override,
                           orient=tk.HORIZONTAL, length=110)
        ovr_sc.pack(side=tk.LEFT)
        _tip(ovr_sc, "左: 低速（1%）  右: 最大速度（100%）")
        tk.Label(spd_inner, textvariable=self._speed_override,
                 bg=BG_PANEL, fg=ACCENT,
                 font=("Consolas", 10, "bold"), width=4).pack(side=tk.LEFT)
        tk.Label(spd_inner, text="%",
                 bg=BG_PANEL, fg=FG_SUB, font=("", 9)).pack(side=tk.LEFT)

        # UTool
        tool_lf = ttk.LabelFrame(right_col, text="  UTool（ツール定義）")
        tool_lf.pack(fill=tk.X, pady=2)
        _tip(tool_lf,
             "UTool（ユーザーツール）:\n"
             "ロボットフランジ先端に取り付けたツールの定義です。\n"
             "・FLANGE: ツールなし（フランジ基準）\n"
             "・KNIFE: 包丁（Z方向に200mmオフセット）\n"
             "3Dビューのシアン色マーカー（TCP）の位置に反映されます。\n"
             "ロボット メニューから数値編集もできます。")
        self._utool_var = tk.StringVar(value=self._active_tool.name)
        tool_names = [f"UT{t.number}: {t.name}  (z={t.z:.0f}mm)" for t in self.TOOL_FRAMES]
        self._utool_combo = ttk.Combobox(tool_lf, textvariable=self._utool_var,
                                          values=tool_names, state="readonly", width=20)
        self._utool_combo.current(1)
        self._utool_combo.pack(padx=6, pady=4)
        self._utool_combo.bind("<<ComboboxSelected>>", self._on_utool_change)

        # UFrame
        uf_lf = ttk.LabelFrame(right_col, text="  UFrame（作業座標系）")
        uf_lf.pack(fill=tk.X, pady=(2, 4))
        _tip(uf_lf,
             "UFrame（ユーザーフレーム）:\n"
             "作業対象（砥石など）の座標系定義です。\n"
             "・UF0 WORLD: ロボット基準座標（デフォルト）\n"
             "・UF9 STONE: 砥石座標系（X=600, Y=25, Z=砥石上面mm, Rz=90°）\n"
             "  kenma 生成の砥石接触座標としても使われます。\n"
             "3Dビューの紫色の座標軸で位置を確認できます。\n"
             "ロボット メニューや UF9 STONE 位置調整パネルから編集できます。")
        self._uframe_var = tk.StringVar(value=self._active_uframe.name)
        uf_names = [f"UF{u.number}: {u.name}" for u in self.USER_FRAMES]
        self._uframe_combo = ttk.Combobox(uf_lf, textvariable=self._uframe_var,
                                           values=uf_names, state="readonly", width=20)
        self._uframe_combo.current(0)
        self._uframe_combo.pack(padx=6, pady=4)
        self._uframe_combo.bind("<<ComboboxSelected>>", self._on_uframe_change)

        # FK 結果（右列下部）
        fk_lf = ttk.LabelFrame(right_col, text="  TCP 位置 / 姿勢 (FK)")
        fk_lf.pack(fill=tk.X, pady=(2, 4))
        _tip(fk_lf,
             "順運動学 (FK: Forward Kinematics)\n"
             "現在の関節角度から計算したTCP（ツール先端）の位置と姿勢です。\n\n"
             "位置 X/Y/Z: ロボット基準座標でのTCP位置 (mm)\n"
             "姿勢 Rx/Ry/Rz: ZYX オイラー角による姿勢 (°)\n\n"
             "スライダーを動かすと即座に更新されます。")
        self._fk_detail_var = tk.StringVar()
        tk.Label(fk_lf, textvariable=self._fk_detail_var,
                 bg=BG_PANEL, fg=ACCENT2,
                 font=("Consolas", 8), anchor="w", justify="left"
                 ).pack(padx=6, pady=4, anchor="w")

        self._update_fk_display()

    def _on_slider_change(self, joint_idx: int, value_deg: float):
        # プログラム由来のスライダー更新（_set_angles / _seek_to_fraction が
        # var.set する分）は、各呼び出し側が明示的に1回だけ再描画するため、
        # ここでの重複再描画・undo記録を抑制する。これを怠ると再生中・
        # ジョグ中に1フレームあたり6回の余分な viewport/FK 再描画が走る。
        if self._seek_updating or self._suppress_undo:
            return
        self._push_undo("スライダー操作", coalesce=True)
        self._joint_angles[joint_idx] = np.deg2rad(value_deg)
        self._update_viewport_from_angles(self._joint_angles)
        self._update_fk_display()

    def _on_utool_change(self, event=None):
        self._push_undo("UTool切替")
        idx = self._utool_combo.current()
        self._active_tool = self.TOOL_FRAMES[idx]
        self.viewport.set_tool_frame(self._active_tool)
        self._set_status(f"✔  UTool 変更 → {self._active_tool.name}  "
                         f"(TCP オフセット: Z={self._active_tool.z:.0f}mm)")

    def _on_uframe_change(self, event=None):
        self._push_undo("UFrame切替")
        idx = self._uframe_combo.current()
        self._active_uframe = self.USER_FRAMES[idx]
        self.viewport.set_user_frame(self._active_uframe)
        self._set_status(f"✔  UFrame 変更 → {self._active_uframe.name}")

    def _build_overlay_panel(self, parent):
        """STL/CSV overlay position control — independent per layer."""
        lf = ttk.LabelFrame(parent, text="  🪨 オーバーレイ位置 (STL / CSV Pose)")
        lf.pack(fill=tk.X, padx=4, pady=(2, 2))
        _tip(lf, "STL・CSV それぞれの位置・姿勢を独立して調整できます。\n"
                 "X/Y/Z: 位置 (mm)  Rx/Ry/Rz: 姿勢 (°)\n"
                 "入力欄でマウスホイール → リアルタイム反映\n"
                 "Ctrl+ホイール=10mm, Shift+ホイール=0.1mm, 通常=1mm")

        self._stl_pose_vars: list = []
        self._csv_pose_vars: list = []

        # ── STL セクション ──────────────────────────────────────
        sf_stl = ttk.LabelFrame(lf, text="  🔵 STL")
        sf_stl.pack(fill=tk.X, padx=4, pady=3)
        inner_stl = ttk.Frame(sf_stl)
        inner_stl.pack(fill=tk.X, padx=4, pady=2)
        for i, axis in enumerate(["X", "Y", "Z", "Rx", "Ry", "Rz"]):
            r, c = divmod(i, 3)
            tk.Label(inner_stl, text=axis, bg=BG_PANEL, fg=FG_SUB,
                     font=("", 8), width=3).grid(row=r, column=c*2, padx=1)
            v = tk.StringVar(value="0.0")
            self._stl_pose_vars.append(v)
            ent = self._make_num_field(inner_stl, v,
                                       on_change=self._apply_stl_pose, width=6)
            ent.grid(row=r, column=c*2+1, padx=1, pady=1)
        btn_stl = ttk.Frame(sf_stl)
        btn_stl.pack(padx=4, pady=(0, 3))
        ttk.Button(btn_stl, text="適用", style="Primary.TButton",
                   command=self._apply_stl_pose).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_stl, text="クリア",
                   command=self._clear_stl).pack(side=tk.LEFT, padx=2)

        # ── CSV セクション ──────────────────────────────────────
        sf_csv = ttk.LabelFrame(lf, text="  🟠 CSV")
        sf_csv.pack(fill=tk.X, padx=4, pady=3)
        inner_csv = ttk.Frame(sf_csv)
        inner_csv.pack(fill=tk.X, padx=4, pady=2)
        for i, axis in enumerate(["X", "Y", "Z", "Rx", "Ry", "Rz"]):
            r, c = divmod(i, 3)
            tk.Label(inner_csv, text=axis, bg=BG_PANEL, fg=FG_SUB,
                     font=("", 8), width=3).grid(row=r, column=c*2, padx=1)
            v = tk.StringVar(value="0.0")
            self._csv_pose_vars.append(v)
            ent = self._make_num_field(inner_csv, v,
                                       on_change=self._apply_csv_pose, width=6)
            ent.grid(row=r, column=c*2+1, padx=1, pady=1)
        btn_csv = ttk.Frame(sf_csv)
        btn_csv.pack(padx=4, pady=(0, 3))
        ttk.Button(btn_csv, text="適用", style="Primary.TButton",
                   command=self._apply_csv_pose).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_csv, text="クリア",
                   command=self._clear_csv).pack(side=tk.LEFT, padx=2)

        # ── 刃先CSV セクション（フランジ追従） ──────────────────────
        sf_blade = ttk.LabelFrame(lf, text="  🔪 刃先CSV（包丁に追従）")
        sf_blade.pack(fill=tk.X, padx=4, pady=3)
        _tip(sf_blade,
             "刃先CSV (x,y,z,nx,ny,nz 形式) を包丁に取り付けます。\n"
             "X/Y/Z/Rx/Ry/Rz: フランジから刃先原点へのオフセット\n"
             "デフォルト: Z=150mm (柄の先端), Ry=180° Rz=-90° (刃を砥石方向へ整列)\n"
             "「再読込」: 同じCSVファイルを再読込（CSV更新時に使用）")
        inner_blade = ttk.Frame(sf_blade)
        inner_blade.pack(fill=tk.X, padx=4, pady=2)
        self._blade_pose_vars: list = []
        blade_defaults = ["0.0", "0.0", "150.0", "0.0", "180.0", "-90.0"]
        for i, axis in enumerate(["X", "Y", "Z", "Rx", "Ry", "Rz"]):
            r, c = divmod(i, 3)
            tk.Label(inner_blade, text=axis, bg=BG_PANEL, fg=FG_SUB,
                     font=("", 8), width=3).grid(row=r, column=c*2, padx=1)
            v = tk.StringVar(value=blade_defaults[i])
            self._blade_pose_vars.append(v)
            ent = self._make_num_field(inner_blade, v,
                                       on_change=self._apply_blade_pose, width=6)
            ent.grid(row=r, column=c*2+1, padx=1, pady=1)
        btn_blade = ttk.Frame(sf_blade)
        btn_blade.pack(padx=4, pady=(0, 3))
        ttk.Button(btn_blade, text="適用", style="Primary.TButton",
                   command=self._apply_blade_pose).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_blade, text="🔄 再読込",
                   command=self._reload_blade_csv).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_blade, text="クリア",
                   command=self._clear_blade).pack(side=tk.LEFT, padx=2)

    # ──────────────────────────────────────────────────────────────────
    # 汎用数値入力欄（マウスホイールで増減・即時反映）
    # ──────────────────────────────────────────────────────────────────

    def _make_num_field(self, parent, var, on_change=None,
                        step=1.0, fine=0.1, coarse=10.0, width=7, fmt="{:.2f}"):
        """ttk.Entry を生成し、マウスホイールで値を増減する数値入力欄を返す。

        ・ホイール上=増加 / 下=減少（Windows delta / Linux Button-4,5 両対応）
        ・Shift=fine（微調整）, Ctrl=coarse（粗調整）, 通常=step
        ・値変更（ホイール）後と <Return> / <FocusOut> で on_change() を呼ぶ
        ・on_change は 3D ビューへの即時反映などに使う（None 可）
        """
        ent = ttk.Entry(parent, textvariable=var, width=width)

        def _amount(event):
            ctrl  = bool(event.state & 0x4)
            shift = bool(event.state & 0x1)
            return coarse if ctrl else (fine if shift else step)

        def _direction(event):
            if getattr(event, "delta", 0):
                return 1 if event.delta > 0 else -1
            return 1 if getattr(event, "num", 0) == 4 else -1

        def _on_wheel(event):
            try:
                current = float(var.get())
            except (ValueError, tk.TclError):
                current = 0.0
            new_val = current + _direction(event) * _amount(event)
            var.set(fmt.format(new_val))
            if on_change is not None:
                on_change()
            return "break"

        def _on_commit(event=None):
            if on_change is not None:
                on_change()

        ent.bind("<MouseWheel>", _on_wheel)
        ent.bind("<Button-4>",   _on_wheel)
        ent.bind("<Button-5>",   _on_wheel)
        ent.bind("<Return>",     _on_commit)
        ent.bind("<FocusOut>",   _on_commit)
        return ent

    # ── ダイアログ用の共通行ヘルパー（セクション見出し / 数値入力行） ──

    @staticmethod
    def _dialog_section(frame, text):
        """ダイアログ内のセクション見出しラベルを追加する。"""
        tk.Label(frame, text=text, bg=BG_PANEL, fg=ACCENT2,
                 font=("Yu Gothic UI", 8, "bold")).pack(anchor="w", pady=(8, 2))

    def _dialog_num_row(self, frame, label, default, hint="",
                        label_width=26, fmt="{:.2f}"):
        """ラベル + 数値入力欄（_make_num_field）+ ヒントの1行を追加する。

        Returns: 入力値の tk.StringVar。
        """
        f = ttk.Frame(frame)
        f.pack(fill=tk.X, pady=2)
        tk.Label(f, text=label, bg=BG_PANEL, fg=FG_PRIMARY,
                 font=("", 8), width=label_width, anchor="w").pack(side=tk.LEFT)
        var = tk.StringVar(value=str(default))
        self._make_num_field(f, var, width=8, fmt=fmt).pack(side=tk.LEFT)
        if hint:
            tk.Label(f, text=hint, bg=BG_PANEL, fg=FG_SUB,
                     font=("", 7)).pack(side=tk.LEFT, padx=4)
        return var

    def _apply_stl_pose(self):
        try:
            vals = [float(v.get()) for v in self._stl_pose_vars]
        except (ValueError, tk.TclError):
            self._set_status("⚠  数値を入力してください")
            return
        self._push_undo("STL位置調整", coalesce=True)
        self.viewport.set_stl_pose(*vals)
        self._set_status(
            f"✔  STL 位置更新: X={vals[0]:.1f} Y={vals[1]:.1f} Z={vals[2]:.1f}")

    def _apply_csv_pose(self):
        try:
            vals = [float(v.get()) for v in self._csv_pose_vars]
        except (ValueError, tk.TclError):
            self._set_status("⚠  数値を入力してください")
            return
        self._push_undo("CSV位置調整", coalesce=True)
        self.viewport.set_csv_pose(*vals)
        self._set_status(
            f"✔  CSV 位置更新: X={vals[0]:.1f} Y={vals[1]:.1f} Z={vals[2]:.1f}")

    def _clear_stl(self):
        self._push_undo("STLクリア")
        self.viewport.clear_stl()
        for v in self._stl_pose_vars:
            v.set("0.0")
        self._set_status("✔  STL オーバーレイをクリアしました")

    def _clear_csv(self):
        self._push_undo("CSVクリア")
        self.viewport.clear_csv()
        for v in self._csv_pose_vars:
            v.set("0.0")
        self._set_status("✔  CSV オーバーレイをクリアしました")

    # ── 刃先CSV ──────────────────────────────────────────────────────

    def _apply_blade_pose(self):
        try:
            vals = [float(v.get()) for v in self._blade_pose_vars]
        except (ValueError, tk.TclError):
            self._set_status("⚠  数値を入力してください")
            return
        self._push_undo("刃先CSV取付調整", coalesce=True)
        self.viewport.set_blade_pose(*vals)
        self._set_status(
            f"✔  刃先CSV 取付位置更新: Z={vals[2]:.1f} Rx={vals[3]:.1f}°")

    def _clear_blade(self):
        self._push_undo("刃先CSVクリア")
        self.viewport.clear_blade()
        self._blade_csv_path = None
        if hasattr(self, "_tree"):
            self._tree_refresh()
        self._set_status("✔  刃先CSV をクリアしました")

    def _load_blade_csv(self, path: Optional[str] = None):
        """刃先CSV (x,y,z,nx,ny,nz) を読み込んで包丁に取り付ける。"""
        if path is None:
            path = filedialog.askopenfilename(
                title="刃先CSV を開く (x,y,z,nx,ny,nz 形式)",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
            if not path:
                return
        with self._undo_group("刃先CSV読込"):
            n = self.viewport.load_blade_csv(path)
            if n == 0:
                messagebox.showerror("読込エラー",
                    "刃先CSVの読込に失敗しました。\n"
                    "形式: x,y,z,nx,ny,nz（6列・ヘッダーなし）")
                return
            self._blade_csv_path = path
            self._apply_blade_pose()  # 現在の取付オフセットを適用
        if hasattr(self, "_tree"):
            self._tree_refresh()
        self._set_status(
            f"✔  刃先CSV 読込: {os.path.basename(path)}  {n} 点 — 包丁に追従表示中")

    def _reload_blade_csv(self):
        """前回読み込んだ刃先CSVを再読込する（CSV更新ワークフロー用）。"""
        path = getattr(self, "_blade_csv_path", None)
        if not path or not os.path.exists(path):
            self._load_blade_csv()
            return
        self._load_blade_csv(path)

    @staticmethod
    def _is_blade_csv(path: str) -> bool:
        """先頭行が6個以上の数値のみなら刃先CSV形式と判定する。"""
        try:
            with open(path, encoding="utf-8-sig") as f:
                first = f.readline().strip()
            fields = first.split(",")
            if len(fields) < 6:
                return False
            for v in fields[:6]:
                float(v)
            return True
        except (ValueError, OSError):
            return False

    def _workflow_load_csv(self):
        """ワークフロー先頭: CSVを開き、刃先CSV/経路CSVを自動判別して読み込む。"""
        path = filedialog.askopenfilename(
            title="CSV を開く（刃先CSV または 経路CSV を自動判別）",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        if self._is_blade_csv(path):
            self._load_blade_csv(path)
        else:
            try:
                loaded = RouteCSVIO.route_from_csv(path)
                self._push_undo("経路CSV読込")
                self.route.waypoints = loaded.waypoints
                self.route.name      = loaded.name
                self.route.comment   = loaded.comment
                self.route_editor.set_route(self.route)
                self._on_route_changed()
                self._set_status(
                    f"✔  経路CSV読込: {len(self.route)} 点 ← {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("読込エラー", f"CSV 読込に失敗しました:\n{e}")

    # ──────────────────────────────────────────────────────────────────
    # RoboDK風 ツリーパネル
    # ──────────────────────────────────────────────────────────────────

    def _build_tree_panel(self, parent):
        """Station ツリーパネルを構築する。"""
        # タイトルは LabelFrame の枠タイトルに頼らず、内側の太字ラベルで
        # 確実に表示する（上端での見切れを防止）。
        lf = ttk.Frame(parent)
        lf.pack(fill=tk.BOTH, expand=True, padx=2, pady=(6, 2))

        tk.Label(lf, text="ステーション", bg=BG_PANEL, fg=ACCENT2,
                 font=("Yu Gothic UI", 10, "bold"), anchor="w").pack(
                     fill=tk.X, padx=4, pady=(4, 2))

        tree_frame = ttk.Frame(lf)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        sb = tk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._tree = ttk.Treeview(tree_frame, show="tree", selectmode="browse",
                                   yscrollcommand=sb.set)
        self._tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        sb.config(command=self._tree.yview)

        # RoboDK 風の色分け（ターゲット=緑 / CSVステップ=青 / 再生=緑太字）
        _tree_font = ("Yu Gothic UI", 9)
        try:
            self._tree.tag_configure("nt_target",  foreground="#7EE787", font=_tree_font)
            self._tree.tag_configure("seq_route",  foreground=ACCENT2,   font=_tree_font)
            self._tree.tag_configure("seq_movej",  foreground="#7EE787", font=_tree_font)
            self._tree.tag_configure("seq_movel",  foreground="#79C0FF", font=_tree_font)
            self._tree.tag_configure("seq_play",
                                     foreground=OK_GREEN,
                                     font=("Yu Gothic UI", 9, "bold"))
            self._tree.tag_configure("drop_target", background=BTN_PRIMARY)
        except tk.TclError:
            pass

        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self._tree.bind("<Double-Button-1>",  self._on_tree_double_click)
        self._tree.bind("<Button-3>",          self._on_tree_right_click)
        self._tree.bind("<F2>",                self._on_tree_rename_key)
        self._tree.bind("<Delete>",            self._on_tree_delete_key)
        # ドラッグ＆ドロップで wp_X / seq_X ノードを並べ替え（Named Target も投入可）
        self._tree_drag_iid: Optional[str] = None
        self._tree_drop_iid: Optional[str] = None
        self._tree.bind("<ButtonPress-1>",   self._on_tree_drag_start)
        self._tree.bind("<B1-Motion>",       self._on_tree_drag_motion)
        self._tree.bind("<ButtonRelease-1>", self._on_tree_drag_release)

        self._tree_refresh()

    def _tree_refresh(self):
        """ツリーを現在の状態で再描画する。"""
        tree = self._tree
        for item in tree.get_children():
            tree.delete(item)

        station = tree.insert("", "end", iid="station",
                               text="📁 Station: 2026_robot", open=True)
        robot   = tree.insert(station, "end", iid="robot",
                               text="🤖 FANUC LR Mate 200iD/14L", open=True)

        # Frames
        frames = tree.insert(robot, "end", iid="frames",
                              text="🔧 Frames", open=True)
        for uf in self.USER_FRAMES:
            # ツリー幅に合わせた短い表示（UF9 STONE は常に USER_FRAMES に含まれる）
            tree.insert(frames, "end", iid=f"uf_{uf.number}",
                         text=f"UF{uf.number}: {uf.name}"
                              f" ({uf.x:.0f},{uf.y:.0f},{uf.z:.0f})")

        # Tools
        tools_node = tree.insert(robot, "end", iid="tools",
                                  text="🔨 Tools", open=True)
        for tf in self.TOOL_FRAMES:
            tree.insert(tools_node, "end", iid=f"ut_{tf.number}",
                         text=f"UT{tf.number}: {tf.name} (z={tf.z:.0f}mm)")
        # 刃先CSV（包丁に追従中の点群）— viewport 未構築時はスキップ
        if hasattr(self, "viewport") and self.viewport.has_blade():
            n_blade = len(self.viewport._blade_pts)
            name = os.path.basename(self._blade_csv_path or "blade.csv")
            tree.insert(tools_node, "end", iid="blade_csv",
                         text=f"[刃先] {name} ({n_blade}pt)")

        # Named Targets ライブラリ（RoboDK のターゲット相当）
        nt_count = len(self._named_targets)
        nt_node = tree.insert(robot, "end", iid="named_targets",
                               text=f"📍 Targets  ({nt_count})",
                               open=True)
        for i, nt in enumerate(self._named_targets):
            tree.insert(nt_node, "end", iid=f"nt_{i}",
                         text=f"⌖ {nt.name}   "
                              f"[{nt.x:.0f}, {nt.y:.0f}, {nt.z:.0f}]",
                         tags=("nt_target",))

        # Route Sequence（CSVルートと Target を自由に組み合わせるプログラム）
        _seq_nums = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
        seq_count = len(self._route_sequence)
        n_pts, length_mm, time_s = self._sequence_stats()
        if seq_count:
            seq_summary = f"  —  {n_pts}点  {length_mm:.0f}mm  {time_s:.1f}s"
        else:
            seq_summary = ""
        seq_node = tree.insert(robot, "end", iid="route_sequence",
                                text=f"📋 Route Sequence  ({seq_count} steps){seq_summary}",
                                open=True)
        for i, si in enumerate(self._route_sequence):
            num = _seq_nums[i] if i < len(_seq_nums) else f"({i+1})"
            if si["type"] == "route":
                r = si["route"]
                n_wps = len(r.waypoints)
                tree.insert(seq_node, "end", iid=f"seq_{i}",
                             text=f"{num}  CSV ▸ {r.name}   ({n_wps}点)",
                             tags=("seq_route",))
            else:
                nt = si["target"]
                mt = si.get("motion", "J")
                spd = si.get("speed", 100 if mt == "J" else 50)
                move = "MoveJ" if mt == "J" else "MoveL"
                unit = "%" if mt == "J" else "mm/s"
                tag = "seq_movej" if mt == "J" else "seq_movel"
                tree.insert(seq_node, "end", iid=f"seq_{i}",
                             text=f"{num}  {move} ▸ {nt.name}   "
                                  f"@{spd:.0f}{unit}",
                             tags=(tag,))
        if self._route_sequence:
            tree.insert(seq_node, "end", iid="seq_play",
                         text="▶  Sequence を再生",
                         tags=("seq_play",))

        # Targets (経路点)
        # 大規模ルート（>25点）はデフォルト折りたたみ＋サマリー表示
        n = len(self.route.waypoints)
        big_route = n > 25
        try:
            _len_mm = self.route.total_length_mm()
            _sec    = self.route.estimated_time_sec()
            _summary = f"  {_len_mm:.0f}mm  {_sec:.1f}s"
        except Exception:
            _summary = ""
        targets = tree.insert(robot, "end", iid="targets",
                               text=f"🎯 Targets  ({n}点){_summary}",
                               open=not big_route)   # 大規模は折りたたみ
        shown = self.route.waypoints[:5] if big_route else self.route.waypoints
        for i, wp in enumerate(shown):
            lbl = wp.label or f"P[{i+1}]"
            tree.insert(targets, "end", iid=f"wp_{i}",
                         text=f"● {lbl}",
                         tags=("waypoint",))
        if big_route:
            tree.insert(targets, "end", iid="targets_more",
                         text=f"… 他{n - 5}点（クリックで展開 / 大規模ルートは省略表示）")

        # Programs (読み込み済み LS)
        progs = tree.insert(robot, "end", iid="programs",
                             text=f"📋 Programs  ({len(self._tree_programs)})",
                             open=True)
        for i, (name, _) in enumerate(self._tree_programs):
            tree.insert(progs, "end", iid=f"prog_{i}", text=f"📄 {name}")

    def _on_tree_select(self, event=None):
        sel = self._tree.selection()
        if not sel:
            return
        iid = sel[0]
        if iid.startswith("wp_"):
            idx = int(iid[3:])
            if 0 <= idx < len(self.route.waypoints):
                self.viewport.set_selected_waypoint(idx)
                self._ik_wp_var.set(idx + 1)

    def _on_tree_rename_key(self, event=None):
        """F2: 選択中の Named Target をリネームする。"""
        sel = self._tree.selection()
        if not sel:
            return
        iid = sel[0]
        if iid.startswith("nt_"):
            self._rename_named_target(int(iid[3:]))
        return "break"

    def _on_tree_delete_key(self, event=None):
        """Delete: 選択中の Target / Sequence ステップ / 経路点を削除する。"""
        sel = self._tree.selection()
        if not sel:
            return
        iid = sel[0]
        if iid.startswith("nt_"):
            self._delete_named_target(int(iid[3:]))
        elif iid.startswith("seq_") and iid != "seq_play":
            self._remove_sequence_item(int(iid[4:]))
        elif iid.startswith("wp_"):
            self._tree_delete_wp(int(iid[3:]))
        return "break"

    def _on_tree_double_click(self, event=None):
        """ダブルクリックで各ノードの主編集アクションを開く。

        ・経路点 (wp_*)    : 従来どおり IK 移動
        ・UFrame (uf_*)    : ユーザーフレーム編集ダイアログ
        ・UTool (ut_*)     : ツールフレーム編集ダイアログ
        ・刃先CSV          : 再読込（右クリックメニューの主アクション）
        ・Programs         : LS ファイル読込
        ・prog_* (各LS)    : 経路に適用
        """
        sel = self._tree.selection()
        if not sel:
            return
        iid = sel[0]
        if iid == "named_targets":
            self._create_named_target_dialog()
            return "break"
        if iid.startswith("nt_"):
            # RoboDK 同様、ダブルクリックでロボットをターゲットへ移動（F2 で改名）
            idx = int(iid[3:])
            self._goto_named_target(idx)
            return "break"
        if iid == "seq_play":
            self._play_sequence()
            return "break"
        if iid.startswith("seq_") and iid != "seq_play":
            idx = int(iid[4:])
            if 0 <= idx < len(self._route_sequence):
                si = self._route_sequence[idx]
                if si["type"] == "route":
                    self._apply_sequence_route(idx)
            return "break"
        if iid.startswith("wp_"):
            idx = int(iid[3:])
            if 0 <= idx < len(self.route.waypoints):
                self._ik_wp_var.set(idx + 1)
                self._compute_ik_for_wp()
            return "break"
        if iid.startswith("uf_"):
            num = int(iid[3:])
            uf = next((u for u in self.USER_FRAMES if u.number == num), None)
            if uf is not None:
                self._edit_user_frame(uf)
            return "break"
        if iid.startswith("ut_"):
            num = int(iid[3:])
            tf = next((t for t in self.TOOL_FRAMES if t.number == num), None)
            if tf is not None:
                self._edit_tool_frame(tf)
            return "break"
        if iid == "blade_csv":
            self._reload_blade_csv()
            return "break"
        if iid == "programs":
            self._load_ls_file()
            return "break"
        if iid.startswith("prog_"):
            i = int(iid[5:])
            if 0 <= i < len(self._tree_programs):
                _, route = self._tree_programs[i]
                self._apply_prog_route(route)
            return "break"

    def _on_tree_right_click(self, event=None):
        item = self._tree.identify_row(event.y)
        if not item:
            return
        self._tree.selection_set(item)

        menu = tk.Menu(self.root, tearoff=0,
                       bg=BG_PANEL, fg=FG_PRIMARY,
                       activebackground=BTN_PRIMARY, activeforeground="white",
                       borderwidth=1, relief="solid")

        if item.startswith("uf_"):
            num = int(item[3:])
            if num == 9:
                menu.add_command(
                    label="🏗  UF9 STONE を自動設定 (x600,y25,STL Z)",
                    command=self._setup_stone_uframe)
                menu.add_separator()
            uf = next((u for u in self.USER_FRAMES if u.number == num), None)
            menu.add_command(
                label="📐  UFrame 編集...（ダブルクリックでも可）",
                command=lambda u=uf: self._edit_user_frame(u))
        elif item.startswith("ut_"):
            num = int(item[3:])
            tf = next((t for t in self.TOOL_FRAMES if t.number == num), None)
            menu.add_command(
                label="🔑  UTool 編集...（ダブルクリックでも可）",
                command=lambda t=tf: self._edit_tool_frame(t))
        elif item == "blade_csv":
            menu.add_command(label="🔄  刃先CSV を再読込",
                             command=self._reload_blade_csv)
            menu.add_command(label="📂  別の刃先CSV を読込...",
                             command=lambda: self._load_blade_csv())
            menu.add_command(label="🗑  刃先CSV をクリア",
                             command=self._clear_blade)
        elif item == "tools":
            menu.add_command(label="📂  刃先CSV を読込...",
                             command=lambda: self._load_blade_csv())
        elif item.startswith("wp_"):
            idx = int(item[3:])
            menu.add_command(label=f"🎯  P[{idx+1}] へ IK 移動",
                             command=lambda i=idx: self._tree_goto_wp(i))
            menu.add_command(label="🗑  この経路点を削除",
                             command=lambda i=idx: self._tree_delete_wp(i))
        elif item == "named_targets":
            menu.add_command(label="📍  現在のTCPをターゲットとして追加...",
                             command=self._create_named_target_dialog)
        elif item.startswith("nt_"):
            i = int(item[3:])
            if 0 <= i < len(self._named_targets):
                nt = self._named_targets[i]
                menu.add_command(label=f"⌖  {nt.name} へ移動（ダブルクリック）",
                                 command=lambda idx=i: self._goto_named_target(idx))
                menu.add_command(label="📝  Teach — 現在のTCP位置で上書き",
                                 command=lambda idx=i: self._teach_named_target(idx))
                menu.add_separator()
                menu.add_command(label="＋  MoveJ で Sequence に追加",
                                 command=lambda idx=i: self._add_target_to_sequence(idx, "J"))
                menu.add_command(label="＋  MoveL で Sequence に追加",
                                 command=lambda idx=i: self._add_target_to_sequence(idx, "L"))
                menu.add_separator()
                menu.add_command(label="✏  名前を変更...（F2）",
                                 command=lambda idx=i: self._rename_named_target(idx))
                menu.add_command(label="🗑  削除（Delete）",
                                 command=lambda idx=i: self._delete_named_target(idx))
        elif item == "route_sequence":
            menu.add_command(label="📋  現在の経路を Sequence に追加",
                             command=self._add_current_route_to_sequence)
            if self._named_targets:
                tsub = tk.Menu(menu, tearoff=0, bg=BG_PANEL, fg=FG_PRIMARY,
                               activebackground=BTN_PRIMARY, activeforeground="white")
                for ti, nt in enumerate(self._named_targets):
                    tsub.add_command(label=f"⌖ {nt.name}",
                                     command=lambda idx=ti: self._add_target_to_sequence(idx))
                menu.add_cascade(label="📍  Target を Sequence に追加", menu=tsub)
            if self._route_sequence:
                menu.add_separator()
                menu.add_command(label="▶  Sequence を再生",
                                 command=self._play_sequence)
                menu.add_command(label="🗑  Sequence をクリア",
                                 command=self._clear_sequence)
        elif item.startswith("seq_") and item != "seq_play":
            i = int(item[4:])
            if 0 <= i < len(self._route_sequence):
                si = self._route_sequence[i]
                menu.add_command(label=f"▶  ここから再生（ステップ{i+1}〜）",
                                 command=lambda idx=i: self._play_sequence(from_idx=idx))
                menu.add_separator()
                if si["type"] == "route":
                    menu.add_command(label="📋  この経路を現在の経路として適用",
                                     command=lambda idx=i: self._apply_sequence_route(idx))
                else:
                    cur = si.get("motion", "J")
                    if cur == "J":
                        menu.add_command(label="🔁  MoveL に変更",
                                         command=lambda idx=i: self._set_sequence_motion(idx, "L"))
                    else:
                        menu.add_command(label="🔁  MoveJ に変更",
                                         command=lambda idx=i: self._set_sequence_motion(idx, "J"))
                    menu.add_command(label="🏃  速度を設定...",
                                     command=lambda idx=i: self._set_sequence_speed(idx))
                menu.add_separator()
                if i > 0:
                    menu.add_command(label="▲  上へ移動",
                                     command=lambda idx=i: self._move_sequence_item(idx, -1))
                if i < len(self._route_sequence) - 1:
                    menu.add_command(label="▼  下へ移動",
                                     command=lambda idx=i: self._move_sequence_item(idx, +1))
                menu.add_command(label="🗑  このステップを削除（Delete）",
                                 command=lambda idx=i: self._remove_sequence_item(idx))
        elif item == "seq_play":
            menu.add_command(label="▶  Sequence を再生",
                             command=self._play_sequence)
        elif item == "targets":
            menu.add_command(label="📂  CSV から読込...", command=self._load_csv)
            menu.add_command(label="📋  LS ファイル読込...", command=self._load_ls_file)
            menu.add_separator()
            menu.add_command(label="🗑  経路をクリア", command=self._clear_route)
        elif item == "programs":
            menu.add_command(label="📂  FANUC LS ファイルを読込...",
                             command=self._load_ls_file)
        elif item.startswith("prog_"):
            i = int(item[5:])
            if 0 <= i < len(self._tree_programs):
                name, route = self._tree_programs[i]
                menu.add_command(label=f"▶  {name} を経路に適用",
                                 command=lambda r=route: self._apply_prog_route(r))
                menu.add_command(label="🗑  リストから削除",
                                 command=lambda idx=i: self._remove_prog(idx))

        menu.tk_popup(event.x_root, event.y_root)

    def _tree_goto_wp(self, idx: int):
        if 0 <= idx < len(self.route.waypoints):
            self._ik_wp_var.set(idx + 1)
            self._compute_ik_for_wp()

    def _tree_delete_wp(self, idx: int):
        if 0 <= idx < len(self.route.waypoints):
            del self.route.waypoints[idx]
            self.route_editor.set_route(self.route)
            self._on_route_changed()

    # ── ツリー ドラッグ＆ドロップ ────────────────────────────────────
    #   ・wp_X  : 経路点の並べ替え
    #   ・seq_X : Sequence ステップの並べ替え
    #   ・nt_X  : Named Target を Sequence へドロップして追加

    def _clear_drop_indicator(self):
        prev = getattr(self, "_tree_drop_iid", None)
        if prev and self._tree.exists(prev):
            # ドロップ強調を外して元のタグへ戻す（_tree_refresh が正しく再設定）
            cur = list(self._tree.item(prev, "tags"))
            cur = [t for t in cur if t != "drop_target"]
            self._tree.item(prev, tags=cur)
        self._tree_drop_iid = None

    def _set_drop_indicator(self, iid):
        if iid == getattr(self, "_tree_drop_iid", None):
            return
        self._clear_drop_indicator()
        if iid and self._tree.exists(iid):
            cur = list(self._tree.item(iid, "tags"))
            if "drop_target" not in cur:
                cur.append("drop_target")
                self._tree.item(iid, tags=cur)
            self._tree_drop_iid = iid

    def _on_tree_drag_start(self, event):
        iid = self._tree.identify_row(event.y)
        if iid and (iid.startswith("wp_")
                    or iid.startswith("nt_")
                    or (iid.startswith("seq_") and iid != "seq_play")):
            self._tree_drag_iid = iid
            self._tree.config(cursor="fleur")
        else:
            self._tree_drag_iid = None

    def _on_tree_drag_motion(self, event):
        if not self._tree_drag_iid:
            return
        target = self._tree.identify_row(event.y)
        src = self._tree_drag_iid
        if src.startswith("wp_"):
            if target and target.startswith("wp_"):
                self._tree.selection_set(target)
        elif src.startswith("seq_"):
            if target and target.startswith("seq_") and target != "seq_play":
                self._tree.selection_set(target)
        elif src.startswith("nt_"):
            # Target を Sequence ノード / ステップへドロップ可能と示す
            if target and (target == "route_sequence"
                           or (target.startswith("seq_"))):
                self._set_drop_indicator(target)
            else:
                self._clear_drop_indicator()

    def _on_tree_drag_release(self, event):
        self._tree.config(cursor="")
        self._clear_drop_indicator()
        src_iid = self._tree_drag_iid
        self._tree_drag_iid = None
        if not src_iid:
            return

        target = self._tree.identify_row(event.y)

        # Named Target を Sequence へドロップして追加
        if src_iid.startswith("nt_"):
            nt_idx = int(src_iid[3:])
            if target == "route_sequence" or target == "seq_play":
                self._add_target_to_sequence(nt_idx)
            elif target and target.startswith("seq_"):
                # 既存ステップの位置に挿入
                dst_idx = int(target[4:])
                self._add_target_to_sequence(nt_idx, insert_at=dst_idx)
            return

        # seq_ アイテムの並べ替え
        if src_iid.startswith("seq_") and src_iid != "seq_play":
            src_idx = int(src_iid[4:])
            if target and target.startswith("seq_") and target != "seq_play":
                dst_idx = int(target[4:])
                n_seq = len(self._route_sequence)
                if src_idx != dst_idx and 0 <= src_idx < n_seq and 0 <= dst_idx < n_seq:
                    si = self._route_sequence.pop(src_idx)
                    self._route_sequence.insert(dst_idx, si)
                    self._seq_dirty = True
                    self._tree_refresh()
                    new_iid = f"seq_{dst_idx}"
                    if self._tree.exists(new_iid):
                        self._tree.selection_set(new_iid)
            return

        # wp_ アイテムの並べ替え
        src_idx = int(src_iid[3:])
        n = len(self.route.waypoints)

        if target and target.startswith("wp_"):
            dst_idx = int(target[3:])
        elif target == "targets_more":
            dst_idx = n - 1          # 「他XX点」の上にドロップ → 末尾へ
        else:
            self._tree.selection_set(src_iid)
            return

        if src_idx == dst_idx or not (0 <= src_idx < n) or not (0 <= dst_idx < n):
            self._tree.selection_set(src_iid)
            return

        self._push_undo("ウェイポイント並べ替え")
        wp = self.route.waypoints.pop(src_idx)
        self.route.waypoints.insert(dst_idx, wp)
        self.route_editor.set_route(self.route)
        self._on_route_changed()

        # 移動後のアイテムを再選択
        new_iid = f"wp_{dst_idx}"
        if self._tree.exists(new_iid):
            self._tree.selection_set(new_iid)
            self._tree.see(new_iid)

    # ── Named Targets ────────────────────────────────────────────────

    def _create_named_target_dialog(self):
        """現在のTCP位置を Named Target として登録するダイアログを開く。"""
        T = self.kin.forward(self._joint_angles)
        x, y, z, rx, ry, rz = self.kin.transform_to_pose(T)

        win = tk.Toplevel(self.root)
        win.title("Named Target を追加")
        win.geometry("340x220")
        win.resizable(False, False)
        win.configure(bg=BG_PANEL)
        win.transient(self.root)
        win.grab_set()

        frm = tk.Frame(win, bg=BG_PANEL)
        frm.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)
        frm.columnconfigure(1, weight=1)

        tk.Label(frm, text="ターゲット名:", bg=BG_PANEL, fg=FG_PRIMARY,
                 font=("Yu Gothic UI", 9)).grid(row=0, column=0, sticky="w", pady=4)
        name_var = tk.StringVar(value=f"Target {len(self._named_targets) + 1}")
        name_entry = tk.Entry(frm, textvariable=name_var, bg=BG_WIDGET, fg=FG_PRIMARY,
                              insertbackground=FG_PRIMARY, relief="flat",
                              font=("Consolas", 9))
        name_entry.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=4)

        pos_text = (
            f"X: {x:.1f} mm   Rx: {rx:.1f} °\n"
            f"Y: {y:.1f} mm   Ry: {ry:.1f} °\n"
            f"Z: {z:.1f} mm   Rz: {rz:.1f} °"
        )
        tk.Label(frm, text=pos_text, bg=BG_PANEL, fg=FG_SUB,
                 font=("Consolas", 9), justify="left").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 12))

        btn_frm = tk.Frame(frm, bg=BG_PANEL)
        btn_frm.grid(row=2, column=0, columnspan=2, sticky="e")

        def _ok():
            nm = name_var.get().strip() or f"Target {len(self._named_targets) + 1}"
            self._named_targets.append(
                _NamedTarget(name=nm, x=x, y=y, z=z, rx=rx, ry=ry, rz=rz))
            if hasattr(self, "_tree"):
                self._tree_refresh()
            self._set_status(f"✔  Named Target 追加: {nm}")
            win.destroy()

        tk.Button(btn_frm, text="OK", command=_ok,
                  bg=BTN_PRIMARY, fg="white", relief="flat",
                  padx=16, pady=4).pack(side=tk.RIGHT, padx=(8, 0))
        tk.Button(btn_frm, text="キャンセル", command=win.destroy,
                  bg=BG_WIDGET, fg=FG_PRIMARY, relief="flat",
                  padx=8, pady=4).pack(side=tk.RIGHT)

        name_entry.focus_set()
        name_entry.select_range(0, tk.END)
        win.bind("<Return>", lambda _e: _ok())
        win.bind("<Escape>", lambda _e: win.destroy())

    def _rename_named_target(self, idx: int):
        """Named Target の名前を変更するダイアログ。"""
        if not (0 <= idx < len(self._named_targets)):
            return
        nt = self._named_targets[idx]

        win = tk.Toplevel(self.root)
        win.title("ターゲット名を変更")
        win.geometry("300x110")
        win.resizable(False, False)
        win.configure(bg=BG_PANEL)
        win.transient(self.root)
        win.grab_set()

        frm = tk.Frame(win, bg=BG_PANEL)
        frm.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)
        frm.columnconfigure(1, weight=1)

        tk.Label(frm, text="新しい名前:", bg=BG_PANEL, fg=FG_PRIMARY,
                 font=("Yu Gothic UI", 9)).grid(row=0, column=0, sticky="w", pady=4)
        name_var = tk.StringVar(value=nt.name)
        name_entry = tk.Entry(frm, textvariable=name_var, bg=BG_WIDGET, fg=FG_PRIMARY,
                              insertbackground=FG_PRIMARY, relief="flat",
                              font=("Consolas", 9))
        name_entry.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        btn_frm = tk.Frame(frm, bg=BG_PANEL)
        btn_frm.grid(row=1, column=0, columnspan=2, sticky="e", pady=(10, 0))

        def _ok():
            nm = name_var.get().strip()
            if nm:
                nt.name = nm
                if hasattr(self, "_tree"):
                    self._tree_refresh()
            win.destroy()

        tk.Button(btn_frm, text="OK", command=_ok,
                  bg=BTN_PRIMARY, fg="white", relief="flat",
                  padx=16, pady=4).pack(side=tk.RIGHT, padx=(8, 0))
        tk.Button(btn_frm, text="キャンセル", command=win.destroy,
                  bg=BG_WIDGET, fg=FG_PRIMARY, relief="flat",
                  padx=8, pady=4).pack(side=tk.RIGHT)

        name_entry.focus_set()
        name_entry.select_range(0, tk.END)
        win.bind("<Return>", lambda _e: _ok())
        win.bind("<Escape>", lambda _e: win.destroy())

    def _delete_named_target(self, idx: int):
        """Named Target を削除する（Sequence 内の参照も削除）。"""
        if not (0 <= idx < len(self._named_targets)):
            return
        nt = self._named_targets[idx]
        self._route_sequence = [
            si for si in self._route_sequence
            if not (si["type"] == "target" and si["target"] is nt)
        ]
        del self._named_targets[idx]
        if hasattr(self, "_tree"):
            self._tree_refresh()

    def _goto_named_target(self, idx: int):
        """Named Target の位置へ IK 移動する。"""
        if not (0 <= idx < len(self._named_targets)):
            return
        nt = self._named_targets[idx]
        T = nt.as_waypoint().to_transform()
        q, ok = self.kin.inverse(T, q_init=self._joint_angles)
        if ok and q is not None:
            self._push_undo(f"Named Target へ移動: {nt.name}")
            self._set_angles(q)
            self._set_status(f"⌖  {nt.name} へ移動しました")
        else:
            messagebox.showwarning("IK 失敗", f"{nt.name} への IK が解けませんでした。\n"
                                   "ターゲット位置が可動範囲外の可能性があります。")

    def _teach_named_target(self, idx: int):
        """Teach: 現在のTCP姿勢でターゲットを上書きする（RoboDK の Teach 相当）。"""
        if not (0 <= idx < len(self._named_targets)):
            return
        nt = self._named_targets[idx]
        T = self.kin.forward(self._joint_angles)
        x, y, z, rx, ry, rz = self.kin.transform_to_pose(T)
        nt.update_pose(x, y, z, rx, ry, rz)
        if hasattr(self, "_tree"):
            self._tree_refresh()
        self._set_status(
            f"📝  Teach: {nt.name} を現在位置で更新 "
            f"({x:.0f}, {y:.0f}, {z:.0f})")

    # ── Route Sequence ───────────────────────────────────────────────

    def _sequence_stats(self):
        """Sequence を平坦化したときの (点数, 距離mm, 推定時間s) を返す。

        ツリー再描画ごとに呼ばれるため、deepcopy せず参照のみで集計する。
        """
        flat = []  # 参照のみ（読み取り専用集計）
        for si in self._route_sequence:
            if si["type"] == "route":
                flat.extend(si["route"].waypoints)
            else:
                mt = si.get("motion", "J")
                spd = si.get("speed", 100 if mt == "J" else 50)
                motion = MotionType.JOINT if mt == "J" else MotionType.LINEAR
                flat.append(si["target"].as_waypoint(speed=spd, motion=motion))
        if not flat:
            return 0, 0.0, 0.0
        tmp = Route(name="_seq")
        tmp.waypoints = flat
        try:
            return len(flat), tmp.total_length_mm(), tmp.estimated_time_sec()
        except Exception:
            return len(flat), 0.0, 0.0

    def _flatten_sequence(self, from_idx: int = 0):
        """Sequence を 1 本の Waypoint 列へ平坦化する（再生用・deepcopy）。

        from_idx を指定すると、そのステップ以降だけを平坦化する。
        """
        flat = []
        for si in self._route_sequence[from_idx:]:
            if si["type"] == "route":
                flat.extend(copy.deepcopy(si["route"].waypoints))
            else:
                mt = si.get("motion", "J")
                spd = si.get("speed", 100 if mt == "J" else 50)
                motion = MotionType.JOINT if mt == "J" else MotionType.LINEAR
                flat.append(si["target"].as_waypoint(speed=spd, motion=motion))
        return flat

    def _add_target_to_sequence(self, nt_idx: int, motion: str = "J",
                                insert_at: Optional[int] = None):
        """Named Target を MoveJ/MoveL 命令として Route Sequence に追加する。"""
        if not (0 <= nt_idx < len(self._named_targets)):
            return
        nt = self._named_targets[nt_idx]
        step = {"type": "target", "target": nt, "motion": motion,
                "speed": 100 if motion == "J" else 50}
        if insert_at is None or not (0 <= insert_at <= len(self._route_sequence)):
            self._route_sequence.append(step)
        else:
            self._route_sequence.insert(insert_at, step)
        self._seq_dirty = True
        if hasattr(self, "_tree"):
            self._tree_refresh()
        move = "MoveJ" if motion == "J" else "MoveL"
        self._set_status(f"✔  Sequence に追加: {move} → {nt.name}")

    def _set_sequence_motion(self, idx: int, motion: str):
        """Sequence のターゲットステップの動作種別(J/L)を変更する。"""
        if not (0 <= idx < len(self._route_sequence)):
            return
        si = self._route_sequence[idx]
        if si["type"] != "target":
            return
        old = si.get("motion", "J")
        si["motion"] = motion
        # 種別変更時は速度の単位が変わるため既定速度へ
        if old != motion:
            si["speed"] = 100 if motion == "J" else 50
        self._seq_dirty = True
        if hasattr(self, "_tree"):
            self._tree_refresh()

    def _set_sequence_speed(self, idx: int):
        """Sequence のターゲットステップの速度を設定するダイアログ。"""
        if not (0 <= idx < len(self._route_sequence)):
            return
        si = self._route_sequence[idx]
        if si["type"] != "target":
            return
        mt = si.get("motion", "J")
        unit = "%" if mt == "J" else "mm/s"
        cur = si.get("speed", 100 if mt == "J" else 50)

        win = tk.Toplevel(self.root)
        win.title("速度を設定")
        win.geometry("280x110")
        win.resizable(False, False)
        win.configure(bg=BG_PANEL)
        win.transient(self.root)
        win.grab_set()

        frm = tk.Frame(win, bg=BG_PANEL)
        frm.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)
        frm.columnconfigure(1, weight=1)

        tk.Label(frm, text=f"速度 ({unit}):", bg=BG_PANEL, fg=FG_PRIMARY,
                 font=("Yu Gothic UI", 9)).grid(row=0, column=0, sticky="w")
        spd_var = tk.StringVar(value=f"{cur:.0f}")
        ent = tk.Entry(frm, textvariable=spd_var, bg=BG_WIDGET, fg=FG_PRIMARY,
                       insertbackground=FG_PRIMARY, relief="flat",
                       font=("Consolas", 9))
        ent.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        btn_frm = tk.Frame(frm, bg=BG_PANEL)
        btn_frm.grid(row=1, column=0, columnspan=2, sticky="e", pady=(12, 0))

        def _ok():
            try:
                v = float(spd_var.get())
                if v > 0:
                    si["speed"] = v
                    self._seq_dirty = True
                    if hasattr(self, "_tree"):
                        self._tree_refresh()
            except ValueError:
                pass
            win.destroy()

        tk.Button(btn_frm, text="OK", command=_ok, bg=BTN_PRIMARY, fg="white",
                  relief="flat", padx=16, pady=4).pack(side=tk.RIGHT, padx=(8, 0))
        tk.Button(btn_frm, text="キャンセル", command=win.destroy, bg=BG_WIDGET,
                  fg=FG_PRIMARY, relief="flat", padx=8, pady=4).pack(side=tk.RIGHT)
        ent.focus_set(); ent.select_range(0, tk.END)
        win.bind("<Return>", lambda _e: _ok())
        win.bind("<Escape>", lambda _e: win.destroy())

    def _move_sequence_item(self, idx: int, delta: int):
        """Sequence ステップを上下に 1 つ移動する。"""
        n = len(self._route_sequence)
        new = idx + delta
        if not (0 <= idx < n) or not (0 <= new < n):
            return
        si = self._route_sequence.pop(idx)
        self._route_sequence.insert(new, si)
        self._seq_dirty = True
        self._tree_refresh()
        if self._tree.exists(f"seq_{new}"):
            self._tree.selection_set(f"seq_{new}")

    def _add_current_route_to_sequence(self):
        """現在の経路を Route Sequence に追加する。"""
        if not self.route.waypoints:
            messagebox.showwarning("経路点なし", "現在の経路に経路点がありません。")
            return
        route_copy = copy.deepcopy(self.route)
        self._route_sequence.append({"type": "route", "route": route_copy})
        self._seq_dirty = True
        if hasattr(self, "_tree"):
            self._tree_refresh()
        self._set_status(
            f"✔  Sequence に追加: CSV {route_copy.name} ({len(route_copy.waypoints)}点)")

    def _remove_sequence_item(self, idx: int):
        """Route Sequence からステップを削除する。"""
        if 0 <= idx < len(self._route_sequence):
            del self._route_sequence[idx]
            self._seq_dirty = True
            if hasattr(self, "_tree"):
                self._tree_refresh()

    def _clear_sequence(self):
        """Route Sequence を全てクリアする。"""
        if not self._route_sequence:
            return
        if messagebox.askyesno("Sequence をクリア", "Route Sequence を全て削除しますか？"):
            self._route_sequence.clear()
            self._seq_dirty = True
            if hasattr(self, "_tree"):
                self._tree_refresh()

    def _apply_sequence_route(self, seq_idx: int):
        """Sequence 内の CSV ルートを現在の経路として適用する。"""
        if not (0 <= seq_idx < len(self._route_sequence)):
            return
        si = self._route_sequence[seq_idx]
        if si["type"] != "route":
            return
        route = si["route"]
        self._push_undo("Sequence 経路を適用")
        self.route.waypoints = copy.deepcopy(route.waypoints)
        self.route.name = route.name
        self.route_editor.set_route(self.route)
        self._on_route_changed()

    def _play_sequence(self, from_idx: Optional[int] = None):
        """Route Sequence を平坦化して連続再生する。

        from_idx を指定すると、そのステップ以降だけを再生する
        （RoboDK の "Run from here" 相当）。
        """
        if not self._route_sequence:
            messagebox.showwarning("Sequence が空",
                                   "Route Sequence にアイテムを追加してください。\n\n"
                                   "・ Target を右クリック → MoveJ/MoveL で Sequence に追加\n"
                                   "・ Target を Sequence へドラッグ＆ドロップ\n"
                                   "・ Route Sequence を右クリック → 現在の経路を追加")
            return

        if from_idx is None or not (0 <= from_idx < len(self._route_sequence)):
            from_idx = 0

        flat_wps = self._flatten_sequence(from_idx=from_idx)
        if not flat_wps:
            return

        self._push_undo("Route Sequence 再生")
        self.route.waypoints = flat_wps
        self.route.name = "ROUTE_SEQUENCE"
        self.route_editor.set_route(self.route)
        self._on_route_changed()
        self._seq_dirty = False
        label = f"ステップ{from_idx+1}〜" if from_idx > 0 else "全体"
        self._set_status(
            f"⌛  Route Sequence 再生（{label}・{len(flat_wps)}点）"
            " — IK 計算中... 完了後に自動再生します")

        token = self._sim_precompute_token

        def _autoplay():
            if self._sim_precompute_token != token:
                return  # ルートが再度変更された場合はキャンセル
            if self._sim_solutions_ready:
                self._seek_var.set(0.0)
                self._play_from_current()
            else:
                self.root.after(200, _autoplay)

        self.root.after(200, _autoplay)

    # ──────────────────────────────────────────────────────────────────
    # ワークフローバー
    # ──────────────────────────────────────────────────────────────────

    def _build_workflow_bar(self, parent):
        """「CSV読込 →→ シミュ →→ 調整 →→ LS出力」水平ワークフローバー。"""
        # 外枠: 横スクロール可能なキャンバスでボタンが見切れないようにする
        outer = tk.Frame(parent, bg=BG_DARK, height=38)
        outer.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
        outer.pack_propagate(False)

        # スクロール可能な内部キャンバス
        wf_canvas = tk.Canvas(outer, bg=BG_DARK, height=38,
                              highlightthickness=0, bd=0)
        wf_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        wf_hscroll = tk.Scrollbar(outer, orient=tk.HORIZONTAL,
                                   command=wf_canvas.xview)
        wf_canvas.configure(xscrollcommand=wf_hscroll.set)

        inner = tk.Frame(wf_canvas, bg=BG_DARK)
        wf_canvas_win = wf_canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(e):
            wf_canvas.configure(scrollregion=wf_canvas.bbox("all"))
            # ボタンが全部収まるなら スクロールバーを隠す
            if inner.winfo_reqwidth() <= wf_canvas.winfo_width():
                wf_hscroll.pack_forget()
            else:
                wf_hscroll.pack(side=tk.BOTTOM, fill=tk.X)
        inner.bind("<Configure>", _on_inner_configure)

        steps = [
            ("📂 CSV読込",  self._workflow_load_csv,
             "CSVを読み込む（自動判別）\n・刃先CSV (x,y,z,nx,ny,nz): 包丁に追従表示\n・経路CSV (ヘッダー付き): 経路点として読込"),
            ("📐 ルート生成", self._kenma_curve_select_dialog,
             "曲線を選択して研磨ルート生成（RoboDK風）\n3Dビューで曲線をクリックした順に研磨順序を指定し HaL/HaR を生成"),
            ("▶ シミュ",    self._start_simulation,    "シミュレーション実行 (F5)"),
            ("🔧 調整",     self._route_adjust_dialog, "経路点の位置・速度を一括調整"),
            ("📤 LS出力",   self._export_tp,           "FANUC LS ファイルを出力 (Ctrl+E)"),
            ("📥 LS読込",   self._load_ls_file,
             "FANUC .ls ファイルを読み込んでツリーに追加\n複数 /PROG（HaL/HaR 等）も対応"),
        ]

        for i, (label, cmd, tip) in enumerate(steps):
            if i > 0:
                tk.Label(inner, text=" →  ", bg=BG_DARK, fg=ACCENT,
                         font=("Consolas", 9, "bold")).pack(side=tk.LEFT)
            style = "Primary.TButton" if i == 0 else "TButton"
            btn = ttk.Button(inner, text=label, command=cmd, style=style)
            btn.pack(side=tk.LEFT, padx=2, pady=5)
            _tip(btn, tip)
            if i == 0:
                # CSV読込の直後に再読込ボタン（頻繁なCSV差し替えワークフロー用）
                reload_btn = ttk.Button(inner, text="🔄", width=3,
                                        command=self._reload_blade_csv)
                reload_btn.pack(side=tk.LEFT, padx=(0, 0))
                _tip(reload_btn, "前回の刃先CSVを再読込（CSVファイル更新時にワンクリック反映）")

        # ワークフロー末尾に「ルートをクリア」（経路点を全削除）
        tk.Label(inner, text="    ", bg=BG_DARK).pack(side=tk.LEFT)
        clear_btn = ttk.Button(inner, text="🗑 ルートをクリア",
                               command=self._clear_route)
        clear_btn.pack(side=tk.LEFT, padx=2, pady=5)
        _tip(clear_btn, "経路点をすべて削除する")

    # ──────────────────────────────────────────────────────────────────
    # FANUC LS ファイル読込
    # ──────────────────────────────────────────────────────────────────

    def _load_ls_file(self):
        """FANUC .ls ファイルを読み込む（ls_to_route 経由）。"""
        from ..path.ls_parser import ls_to_route as _ls_to_route

        path = filedialog.askopenfilename(
            title="FANUC LS ファイルを開く",
            filetypes=[("FANUC TP", "*.ls *.LS"), ("All files", "*.*")])
        if not path:
            return
        path, err = _validate_path(path, ".ls", "LS ファイル")
        if err:
            messagebox.showerror("LS 読込エラー", err)
            return
        try:
            routes = _ls_to_route(path, self.kin)
        except Exception as e:
            messagebox.showerror("LS 読込エラー", f"読込に失敗しました:\n{e}")
            return

        if not routes:
            messagebox.showwarning("LS 読込", "有効な経路点が見つかりませんでした。")
            return

        base = os.path.basename(path)

        if len(routes) == 1:
            r = routes[0]
            ans = messagebox.askyesno(
                "LS 読込",
                f"プログラム: {r.name}\n経路点数: {len(r.waypoints)} 点\n\n"
                "現在の経路と置き換えますか？\n"
                "「いいえ」→ ツリーに追加のみ",
                icon="question")
            if ans:
                self._apply_prog_route(r)
            else:
                self._push_undo("LSツリー追加")
                self._tree_programs.append((r.name, r))
                self._tree_refresh()
                self._set_status(
                    f"✔  LS 読込 (ツリー追加): {r.name}  {len(r.waypoints)} 点")
        else:
            total = sum(len(r.waypoints) for r in routes)
            names = " / ".join(r.name for r in routes)
            ans = messagebox.askyesnocancel(
                "LS 読込 — 複数プログラム",
                f"検出プログラム数: {len(routes)}\n"
                f"  {names}\n\n"
                f"「はい」: 全プログラムを結合して経路に適用 ({total} 点)\n"
                "「いいえ」: ツリーに追加のみ\n"
                "「キャンセル」: 何もしない",
                icon="question")
            if ans is True:
                merged = routes[0]
                for r in routes[1:]:
                    merged.waypoints.extend(r.waypoints)
                merged.name = os.path.splitext(base)[0]
                self._apply_prog_route(merged)
            elif ans is False:
                self._push_undo("LSツリー追加")
                for r in routes:
                    self._tree_programs.append((r.name, r))
                self._tree_refresh()
                self._set_status(
                    f"✔  LS 読込 (ツリー追加): {len(routes)} プログラム  計 {total} 点")

    def _apply_prog_route(self, route):
        """読み込んだ Route を現在の経路として適用する。"""
        self._push_undo("LS経路適用")
        self.route.waypoints = list(route.waypoints)
        self.route.name      = route.name
        self.route.comment   = route.comment
        if route.uframe:
            self.route.uframe = route.uframe
        if route.utool:
            self.route.utool  = route.utool
        self.route_editor.set_route(self.route)
        self.viewport.set_route(self.route)   # set_route が再描画する
        self._tree_refresh()
        self._invalidate_sim_solutions()
        self._set_status(
            f"✔  経路適用: {route.name}  {len(self.route)} 点")

    def _remove_prog(self, idx: int):
        if 0 <= idx < len(self._tree_programs):
            self._push_undo("プログラム削除")
            del self._tree_programs[idx]
            self._tree_refresh()

    # ──────────────────────────────────────────────────────────────────
    # UF9 STONE 自動設定
    # ──────────────────────────────────────────────────────────────────

    def _stl_world_z(self) -> Optional[np.ndarray]:
        """ワールド変換後の STL 全頂点 Z 座標配列を返す（未読込なら None）。"""
        if self.viewport.stl_bbox() is None or self.viewport._stl_verts is None:
            return None
        R = self.viewport._stl_T[:3, :3]
        t = self.viewport._stl_T[:3, 3]
        all_v = self.viewport._stl_verts.reshape(-1, 3)
        return (R @ all_v.T).T[:, 2] + t[2]

    def _stl_top_z(self, fallback: float) -> float:
        """ワールド変換後の STL 上面 Z [mm] を返す（STL 未読込時は fallback）。"""
        z = self._stl_world_z()
        return float(z.max()) if z is not None else fallback

    def _stl_bottom_z(self, fallback: float) -> float:
        """ワールド変換後の STL 下面 Z [mm] を返す（STL 未読込時は fallback）。"""
        z = self._stl_world_z()
        return float(z.min()) if z is not None else fallback

    def _setup_stone_uframe(self):
        """STL を床 (Z=0) に置き、上面を UF9 STONE Z として自動設定する。"""
        self._push_undo("UF9 STONE 自動設定")
        if self.viewport._stl_verts is not None:
            # 砥石を床に置く（底面 Z=0 に調整）
            stone_bottom_z = self._stl_bottom_z(0.0)
            if abs(stone_bottom_z) > 0.5:
                vals = [float(v.get() or 0) for v in self._stl_pose_vars]
                new_z = vals[2] - stone_bottom_z
                self.viewport.set_stl_pose(vals[0], vals[1], new_z,
                                           vals[3], vals[4], vals[5])
                for var, val in zip(self._stl_pose_vars,
                                    [vals[0], vals[1], new_z,
                                     vals[3], vals[4], vals[5]]):
                    var.set(f"{val:.2f}")
        grinder_top_z = self._stl_top_z(UserFrame.stone9().z)
        self._sync_uf9(self._uf9_frame(z=grinder_top_z))

        # UF9 をアクティブ UFrame に切り替え
        idx9 = next(i for i, uf in enumerate(self.USER_FRAMES) if uf.number == 9)
        self._uframe_combo.current(idx9)
        self._active_uframe = self.USER_FRAMES[idx9]
        self.viewport.set_user_frame(self._active_uframe)
        self._set_status(
            f"✔  UF9 STONE 設定: x=600, y=25, z={grinder_top_z:.0f}mm, rz=90°")

    # ──────────────────────────────────────────────────────────────────
    # 経路調整ダイアログ
    # ──────────────────────────────────────────────────────────────────

    def _route_adjust_dialog(self):
        """経路点を一括調整するダイアログを開く。"""
        if not self.route.waypoints:
            messagebox.showwarning("経路点なし", "経路点がありません。\nまず経路を読み込んでください。")
            return

        win = tk.Toplevel(self.root)
        win.title("経路調整")
        win.geometry("420x390")
        win.configure(bg=BG_DARK)
        win.resizable(False, False)

        tk.Label(win, text="🔧  経路点一括調整",
                 bg=BG_DARK, fg=ACCENT,
                 font=("Yu Gothic UI", 11, "bold")).pack(pady=(12, 2), padx=12, anchor="w")
        tk.Label(win,
                 text=f"対象: {len(self.route.waypoints)} 点  [{self.route.name}]",
                 bg=BG_DARK, fg=FG_SUB,
                 font=("", 8)).pack(padx=12, anchor="w")

        ttk.Separator(win).pack(fill=tk.X, padx=12, pady=8)

        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=16)

        def section(text):
            self._dialog_section(frame, text)

        def row(label, default, hint=""):
            return self._dialog_num_row(frame, label, default, hint,
                                        label_width=22)

        section("▸ 位置オフセット (mm) — 全点に加算")
        v_dx = row("ΔX:", "0", "前後方向シフト")
        v_dy = row("ΔY:", "0", "左右方向シフト")
        v_dz = row("ΔZ:", "0", "上下方向シフト")

        section("▸ 速度調整")
        v_spd = row("速度スケール (%):", "100", "100=変更なし  50=半速")

        section("▸ 経路オプション")
        v_rev = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="経路を逆順にする",
                        variable=v_rev).pack(anchor="w", pady=2)

        def _apply():
            try:
                dx  = float(v_dx.get())
                dy  = float(v_dy.get())
                dz  = float(v_dz.get())
                spd = float(v_spd.get()) / 100.0
            except ValueError:
                messagebox.showerror("入力エラー", "数値を入力してください", parent=win)
                return
            self._push_undo("経路一括調整")
            for wp in self.route.waypoints:
                wp.x += dx
                wp.y += dy
                wp.z += dz
                wp.speed = max(1.0, wp.speed * spd)
            if v_rev.get():
                self.route.waypoints.reverse()
            self.route_editor.set_route(self.route)
            self._on_route_changed()
            self._set_status(
                f"✔  経路調整: ΔX={dx:.1f} ΔY={dy:.1f} ΔZ={dz:.1f}"
                f"  速度×{spd:.2f}"
                f"{'  逆順' if v_rev.get() else ''}")
            win.destroy()

        ttk.Separator(win).pack(fill=tk.X, padx=16, pady=8)
        btn_row = ttk.Frame(win)
        btn_row.pack(pady=4)
        ttk.Button(btn_row, text="✔  適用",
                   style="Primary.TButton",
                   command=_apply).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_row, text="キャンセル",
                   command=win.destroy).pack(side=tk.LEFT, padx=6)

    def _on_viewport_drop(self, event):
        raw = event.data.strip()
        # Windows: path may be wrapped in braces for paths with spaces
        paths = self.root.tk.splitlist(raw)
        if not paths:
            return
        path = paths[0].strip("{}")
        ext = os.path.splitext(path)[1].lower()
        if ext == ".stl":
            self._push_undo("STL読込")
            ok = self.viewport.load_stl(path)
            if ok:
                self._set_status(f"✔  STL 読込: {os.path.basename(path)}")
            else:
                self._set_status("⚠  STL 読込失敗: numpy-stl が必要です (pip install numpy-stl)")
        elif ext == ".csv":
            if self._is_blade_csv(path):
                self._load_blade_csv(path)
            else:
                self._push_undo("CSV読込")
                ok = self.viewport.load_csv_points(path)
                if ok:
                    self._set_status(f"✔  CSV 読込: {os.path.basename(path)}")
                else:
                    self._set_status("⚠  CSV に有効な X,Y,Z 列がありません")
        elif ext == ".json":
            # 設定ファイル（ファイル → 設定をファイルへ出力 で生成）
            self._import_settings(path)
        else:
            self._set_status(f"⚠  対応形式: .stl / .csv / .json（設定） ({ext})")

    # ──────────────────────────────────────────────────────────────────
    # ステータスバー
    # ──────────────────────────────────────────────────────────────────

    def _build_status_bar(self):
        bar = tk.Frame(self.root, bg=BG_DARK, height=24)
        bar.pack(fill=tk.X, side=tk.BOTTOM)

        # 左：ステータスメッセージ
        self._status_var = tk.StringVar(value="準備完了 — ショートカットキー: Ctrl+O 読込 / Ctrl+S 保存 / Ctrl+E TP出力 / F5 実行")
        tk.Label(bar,
            textvariable=self._status_var,
            bg=BG_DARK, fg=FG_SUB,
            font=("Yu Gothic UI", 8), anchor="w"
        ).pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)

        # 右：バージョン
        tk.Label(bar,
            text=f"v{APP_VERSION}",
            bg=BG_DARK, fg=ACCENT,
            font=("Consolas", 8, "bold")
        ).pack(side=tk.RIGHT, padx=8)

        tk.Frame(bar, bg=BORDER, width=1).pack(side=tk.RIGHT, fill=tk.Y, pady=4)

        # 右：ロボット名
        tk.Label(bar,
            text="FANUC LR Mate 200iD/14L",
            bg=BG_DARK, fg=FG_SUB,
            font=("", 8)
        ).pack(side=tk.RIGHT, padx=8)

    # ──────────────────────────────────────────────────────────────────
    # Jog
    # ──────────────────────────────────────────────────────────────────

    def _jog(self, axis: int, direction: int):
        try:
            step = float(self._jog_step.get())
        except ValueError:
            step = 5.0

        if self._jog_mode.get() == "Joint":
            self._push_undo("ジョグ操作", coalesce=True)
            q = self._joint_angles.copy()
            q[axis] += np.deg2rad(step * direction)
            lower, upper = self.kin.dh.get_joint_limits()
            q[axis] = float(np.clip(q[axis], lower[axis], upper[axis]))
            self._set_angles(q)
        else:
            T = self.kin.forward(self._joint_angles)
            if axis < 3:
                T[:3, 3][axis] += step * direction
            else:
                from scipy.spatial.transform import Rotation
                delta_deg = np.zeros(3)
                delta_deg[axis - 3] = step * direction
                dR = Rotation.from_euler("xyz", delta_deg, degrees=True).as_matrix()
                T[:3, :3] = dR @ T[:3, :3]
            q, ok = self.kin.inverse(T, q_init=self._joint_angles)
            if ok:
                # IK 成功時のみ undo を記録（失敗時は状態が変わらないため）
                self._push_undo("ジョグ操作", coalesce=True)
                self._set_angles(q)
                self.viewport.set_jog_target(T[:3, 3])
            else:
                self._set_status("⚠  Cartesian ジョグ: IK 失敗 — 可動範囲外の位置です")

    # ──────────────────────────────────────────────────────────────────
    # Viewport & FK
    # ──────────────────────────────────────────────────────────────────

    def _update_viewport_from_angles(self, q: np.ndarray):
        self.viewport.update_robot(q)

    def _update_fk_display(self):
        T  = self.kin.forward(self._joint_angles)
        x, y, z, rx, ry, rz = self.kin.transform_to_pose(T)
        if hasattr(self, "_fk_detail_var"):
            self._fk_detail_var.set(
                f"  位置 X: {x:8.1f} mm    姿勢 Rx: {rx:7.1f} °\n"
                f"  位置 Y: {y:8.1f} mm    姿勢 Ry: {ry:7.1f} °\n"
                f"  位置 Z: {z:8.1f} mm    姿勢 Rz: {rz:7.1f} °"
            )

    # ──────────────────────────────────────────────────────────────────
    # Route events
    # ──────────────────────────────────────────────────────────────────

    def _on_route_changed(self):
        self.viewport.set_route(self.route)   # set_route が再描画する
        n   = len(self.route)
        self._set_status(f"✔  経路更新 — {n} 点")
        if hasattr(self, "_tree"):
            self._tree_refresh()
        self._invalidate_sim_solutions()

    def _on_waypoint_selected(self, idx: int):
        self.viewport.set_selected_waypoint(idx)

    # ──────────────────────────────────────────────────────────────────
    # File I/O
    # ──────────────────────────────────────────────────────────────────

    def _load_csv(self):
        path = filedialog.askopenfilename(
            title="CSV ファイルを開く",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        path, err = _validate_path(path, ".csv", "CSV ファイル")
        if err:
            messagebox.showerror("読込エラー", err)
            return
        try:
            loaded = RouteCSVIO.route_from_csv(path)
            self._push_undo("経路CSV読込")
            self.route.waypoints = loaded.waypoints
            self.route.name      = loaded.name
            self.route.comment   = loaded.comment
            self.route_editor.set_route(self.route)
            self.viewport.set_route(self.route)   # set_route が再描画する
            self._invalidate_sim_solutions()
            self._set_status(f"✔  読込完了: {len(self.route)} 点 ← {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("読込エラー", f"CSV 読込に失敗しました:\n{e}")

    def _save_csv(self):
        path = filedialog.asksaveasfilename(
            title="CSV として保存", defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{self.route.name}.csv")
        if not path:
            return
        try:
            RouteCSVIO.route_to_csv(self.route, path)
            self._set_status(f"✔  保存完了: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("保存エラー", f"CSV 保存に失敗しました:\n{e}")

    def _export_tp(self):
        if not self.route.waypoints:
            messagebox.showwarning("経路点なし", "経路点が1つもありません。\nまず経路点を追加してください。")
            return

        # UF/UT frame setup dialog for kenma routes
        uframe_num = self.route.uframe or self._active_uframe.number
        utool_num  = self.route.utool  or self._active_tool.number

        # If route uses UF9 (kenma), offer to embed frame definitions
        uframe_pos = None
        utool_pos  = None
        if uframe_num == 9:
            ans = messagebox.askyesno(
                "FANUC LS エクスポート",
                f"UFrame={uframe_num} / UTool={utool_num} を使用します。\n\n"
                "「はい」: UF9/UT9 の座標定義を LS ファイル先頭に追加\n"
                "　　　　　（実機ロボットで正しく動くように）\n\n"
                "「いいえ」: UFRAME_NUM/UTOOL_NUM の設定のみ出力",
                icon="question")
            if ans:
                # UF9: 既定値（UserFrame.stone9 = X=600,Y=25,Z=340,W=0,P=0,R=90）
                s9 = UserFrame.stone9()
                uframe_pos = (s9.x, s9.y, s9.z, s9.rx, s9.ry, s9.rz)
                # UT9: X=0,Y=0,Z=150,W=-90,P=0,R=90
                utool_pos  = (0.0, 0.0, 150.0, -90.0, 0.0, 90.0)

        path = filedialog.asksaveasfilename(
            title="FANUC TP プログラムをエクスポート", defaultextension=".ls",
            filetypes=[("FANUC TP", "*.ls"), ("All files", "*.*")],
            initialfile=f"{self.route.name}.ls")
        if not path:
            return
        self._set_status("IK 計算中 — しばらくお待ちください...")
        self.root.update()
        try:
            exporter = TPExporter(self.kin)
            exporter.export(self.route, path,
                            utool=utool_num,
                            uframe=uframe_num,
                            speed_override=self._speed_override.get(),
                            uframe_pos=uframe_pos,
                            utool_pos=utool_pos)
            self._set_status(f"✔  TP 出力完了: {os.path.basename(path)}")
            with open(path) as f:
                content = f.read()
            self._show_text_preview(f"TP プレビュー: {os.path.basename(path)}", content)
        except Exception as e:
            messagebox.showerror("TP エクスポートエラー", f"エクスポートに失敗しました:\n{e}")

    # ──────────────────────────────────────────────────────────────────
    # シークバー（タイムラインスクラバー）
    # ──────────────────────────────────────────────────────────────────

    def _build_seek_bar(self, parent):
        """YouTube/RoboDK 風のシークバー（タイムラインスクラバー）。"""
        bar = ttk.Frame(parent)
        bar.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=(2, 2))

        # ⏮ 頭出し
        ttk.Button(bar, text="⏮", style="Jog.TButton",
                   command=self._seek_to_start).pack(side=tk.LEFT, padx=1)
        # ▶/⏸ 再生・一時停止トグル
        self._play_btn_var = tk.StringVar(value="▶")
        self._seek_play_btn = ttk.Button(
            bar, textvariable=self._play_btn_var, style="Jog.TButton",
            command=self._toggle_play)
        self._seek_play_btn.pack(side=tk.LEFT, padx=1)
        _tip(self._seek_play_btn, "再生 / 一時停止")
        # ⏹ 停止（先頭へリセット）
        ttk.Button(bar, text="⏹", style="Jog.TButton",
                   command=self._seek_stop).pack(side=tk.LEFT, padx=1)

        # シークバー本体（0..N-1 の分数インデックス）
        self._seek_var = tk.DoubleVar(value=0.0)
        self._seek_scale = ttk.Scale(
            bar, from_=0.0, to=1.0, variable=self._seek_var,
            orient=tk.HORIZONTAL, command=self._on_seek_scale)
        self._seek_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self._seek_scale.bind("<ButtonPress-1>",   self._on_seek_press)
        self._seek_scale.bind("<ButtonRelease-1>", self._on_seek_release)
        _tip(self._seek_scale,
             "ドラッグで経路をスクラブできます（IK事前計算済みで即時表示）")

        # 時間 / 経路点インデックス表示
        self._seek_info_var = tk.StringVar(value="経過 0.0s / 総 0.0s   P[0/0]")
        tk.Label(bar, textvariable=self._seek_info_var,
                 bg=BG_PANEL, fg=ACCENT, font=("Consolas", 8),
                 anchor="e", width=30).pack(side=tk.LEFT, padx=4)

        # 初期は事前計算待ち（無効化）
        self._set_seek_enabled(False)

    def _set_seek_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        if hasattr(self, "_seek_scale"):
            self._seek_scale.config(state=state)
        if hasattr(self, "_seek_play_btn"):
            self._seek_play_btn.config(state=state)

    def _on_pre_fps_change(self):
        """FPS スピンボックス変更時: _pre_fps を更新してフレームキャッシュを無効化。"""
        try:
            v = int(self._pre_fps_var.get())
        except (ValueError, tk.TclError):
            v = 20
        self._pre_fps = float(max(5, min(30, v)))
        self._pre_fps_var.set(int(self._pre_fps))
        if self._pre_frames is not None:
            self._pre_frames = None
            self._pre_idxs = None
            if hasattr(self, "_smooth_btn"):
                self._smooth_btn.config(text="🎬  滑らか再生（事前描画）")
            if hasattr(self, "_save_video_btn"):
                self._save_video_btn.config(state="disabled")

    # ── IK 事前計算（バックグラウンド） ───────────────────────────────

    def _invalidate_sim_solutions(self):
        """ルート変更時に IK 解キャッシュを破棄して再計算を起動する。"""
        self._sim_solutions = []
        self._sim_solutions_ready = False
        # 事前描画キャッシュもルート変更で無効化
        self._pre_frames = None
        self._pre_idxs = None
        if hasattr(self, "_save_video_btn"):
            self._save_video_btn.config(state="disabled")
        if hasattr(self, "_smooth_btn"):
            self._smooth_btn.config(text="🎬  滑らか再生（事前描画）")
        self._sim_precompute_token += 1
        token = self._sim_precompute_token
        self._set_seek_enabled(False)

        n = len(self.route.waypoints)
        if hasattr(self, "_seek_scale"):
            self._seek_scale.config(from_=0.0, to=max(1.0, float(n - 1)))
        if hasattr(self, "_seek_var"):
            self._seek_var.set(0.0)
        if n == 0:
            self._sim_solutions_ready = True
            if hasattr(self, "_seek_info_var"):
                self._seek_info_var.set("経過 0.0s / 総 0.0s   P[0/0]")
            return

        if hasattr(self, "_seek_info_var"):
            self._seek_info_var.set("IK事前計算中...")

        waypoints = list(self.route.waypoints)
        q_seed = self._joint_angles.copy()

        def work():
            def progress_cb(i: int, wp_n: int):
                def _upd(i=i, wp_n=wp_n):
                    if token != self._sim_precompute_token:
                        return
                    if hasattr(self, "_seek_info_var"):
                        self._seek_info_var.set(f"IK計算中 P[{i + 1}/{wp_n}]...")
                self.root.after(0, _upd)

            sols, n_warn = self._compute_sim_solutions(waypoints, q_seed,
                                                       progress_cb=progress_cb)
            def done():
                if token != self._sim_precompute_token:
                    return  # 古い計算結果は破棄
                self._sim_solutions = sols
                self._sim_solutions_ready = True
                self._set_seek_enabled(True)
                self._update_seek_info(0.0)
                # GPU(リアルタイム)版は事前描画キャッシュ無しでも動画保存できる
                # （保存時にオンデマンドで事前描画する）ため、IK完了で有効化する。
                if getattr(self.viewport, "realtime", False) \
                        and len(sols) >= 2 and hasattr(self, "_save_video_btn"):
                    self._save_video_btn.config(state="normal")
                if n_warn > 0:
                    self._set_status(
                        f"⚠  床干渉 {n_warn}点 — 代替 IK でも回避不能。"
                        " 経路点の Z 高さを上げてください。")
            self.root.after(0, done)

        self._sim_precompute_thread = threading.Thread(target=work, daemon=True)
        self._sim_precompute_thread.start()

    def _compute_sim_solutions(self, waypoints, q_seed, progress_cb=None):
        """全経路点の IK を連鎖シードで解く。

        各点で床干渉（関節位置 Z < 0）を検出したら代替 IK シードを並列に試みる。
        戻り値: (sols, n_floor_warn)
          sols: List[np.ndarray] 各経路点の関節解（len == len(waypoints)）
          n_floor_warn: 床干渉が残った経路点数（0 なら安全）
        """
        FLOOR_Z = 0.0        # ロボット取付面 = 床
        MARGIN  = 10.0       # 10mm の余裕を設ける（mm）

        # 床を踏まない代替 IK シード（腕を上方に持ち上げる構成）
        arm_up_seeds = [
            self.kin.dh.ready_position(),
            np.deg2rad([0,  -90,  30,   0, -60,   0]),
            np.deg2rad([0,  -80,  10,   0, -50,   0]),
            np.deg2rad([0,  -70,  20,  90, -60,   0]),
            np.deg2rad([0,  -70,  20, -90, -60,   0]),
        ]

        def _floor_ok(q: np.ndarray) -> bool:
            pts = self.kin.get_joint_positions(q)
            return float(pts[:, 2].min()) >= FLOOR_Z - MARGIN

        sols = []
        n_floor_warn = 0
        q_prev = np.asarray(q_seed, dtype=float).copy()
        n = len(waypoints)

        for i, wp in enumerate(waypoints):
            if progress_cb is not None:
                progress_cb(i, n)

            T = wp.to_transform()
            q, ok = self.kin.inverse(T, q_init=q_prev)
            if not ok or q is None:
                q = q_prev

            # 床干渉なければそのまま使う
            if _floor_ok(q):
                sols.append(q.copy())
                q_prev = q
                continue

            # 代替シードを並列に試して床干渉のない解を探す（scipy は GIL 外で真の並列化）
            best_q     = q
            best_min_z = float(self.kin.get_joint_positions(q)[:, 2].min())
            found_safe = False

            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(arm_up_seeds)) as ex:
                alt_results = list(
                    ex.map(lambda seed: self.kin.inverse(T, q_init=seed),
                           arm_up_seeds))

            for q_alt, ok_alt in alt_results:
                if not ok_alt or q_alt is None:
                    continue
                if _floor_ok(q_alt):
                    best_q = q_alt
                    found_safe = True
                    break
                min_z = float(self.kin.get_joint_positions(q_alt)[:, 2].min())
                if min_z > best_min_z:
                    best_min_z = min_z
                    best_q = q_alt

            if not found_safe:
                n_floor_warn += 1

            sols.append(best_q.copy())
            q_prev = best_q

        return sols, n_floor_warn

    def _segment_times(self):
        """各セグメントの所要時間（速度OVR込み）と累積時間配列を返す。

        戻り値: (cum_time[N], total) — cum_time[i] は P[0]→P[i] までの累積秒。
        """
        wps = self.route.waypoints
        n = len(wps)
        cum = [0.0] * n
        override = max(self._speed_override.get() / 100.0, 1e-6)
        for i in range(1, n):
            d = float(np.linalg.norm(
                np.asarray(wps[i].position()) - np.asarray(wps[i - 1].position())))
            speed = max(wps[i].speed, 1.0)
            cum[i] = cum[i - 1] + (d / speed) / override
        total = cum[-1] if n else 0.0
        return cum, total

    # ── スクラブ（ドラッグ） ───────────────────────────────────────────

    def _on_seek_press(self, event=None):
        self._seek_dragging = True
        # ドラッグ開始時は再生を止める（位置はそのまま）
        self._sim_playing = False
        self._sim_running = False
        self._play_btn_var.set("▶")

    def _on_seek_release(self, event=None):
        self._seek_dragging = False

    def _on_seek_scale(self, value):
        """スライダー値変更（コマンドコールバック）— 即時にロボット姿勢を設定。"""
        if self._seek_updating:
            return  # 自動更新由来は無視（無限ループ防止）
        if not self._sim_solutions_ready or not self._sim_solutions:
            return
        try:
            idx_f = float(value)
        except (ValueError, tk.TclError):
            return
        self._seek_to_fraction(idx_f)

    @staticmethod
    def _time_to_idx(t: float, cum: np.ndarray, total: float, n: int) -> float:
        """単調増加の累積時間配列 cum 上で、時刻 t に対応する分数インデックスを
        線形補間で返す（cum[i-1]〜cum[i] の区間内を按分）。"""
        if t <= 0:
            return 0.0
        if t >= total:
            return float(n - 1)
        # cum[i] >= t となる最初の i を二分探索で求める（O(log n)）。
        i = int(np.searchsorted(cum, t, side="left"))
        i = max(1, min(i, n - 1))
        seg = cum[i] - cum[i - 1]
        if seg <= 1e-9:
            return float(i)
        return (i - 1) + (t - cum[i - 1]) / seg

    def _seek_to_fraction(self, idx_f: float):
        """分数インデックス idx_f の補間姿勢にロボットを設定する（IK計算なし）。"""
        sols = self._sim_solutions
        n = len(sols)
        if n == 0:
            return
        idx_f = max(0.0, min(float(n - 1), idx_f))
        i0 = int(np.floor(idx_f))
        i1 = min(i0 + 1, n - 1)
        alpha = idx_f - i0
        q = sols[i0] + alpha * (sols[i1] - sols[i0])

        self._joint_angles = q.copy()
        self._update_viewport_from_angles(q)
        self._update_fk_display()
        self._seek_updating = True
        try:
            for j, var in enumerate(self._slider_vars):
                var.set(np.rad2deg(q[j]))
        finally:
            self._seek_updating = False
        self.viewport.set_selected_waypoint(i0)
        self._update_seek_info(idx_f)

    def _update_seek_info(self, idx_f: float):
        """経過/総時間と P[i/N] ラベルを更新する。"""
        n = len(self.route.waypoints)
        cum, total = self._segment_times()
        idx_f = max(0.0, min(float(max(n - 1, 0)), idx_f))
        i0 = int(np.floor(idx_f))
        i1 = min(i0 + 1, max(n - 1, 0))
        alpha = idx_f - i0
        if n >= 1 and i0 < len(cum):
            elapsed = cum[i0] + alpha * (cum[i1] - cum[i0]) if i1 < len(cum) else cum[i0]
        else:
            elapsed = 0.0
        self._seek_info_var.set(
            f"経過 {elapsed:.1f}s / 総 {total:.1f}s   P[{i0+1 if n else 0}/{n}]")

    # ── 再生 / 一時停止 / 停止 ─────────────────────────────────────────

    def _toggle_play(self):
        if self._sim_playing:
            self._pause_play()
        else:
            self._play_from_current()

    def _seek_to_start(self):
        """⏮ 頭出し（先頭へ）。"""
        self._pause_play()
        if hasattr(self, "_seek_var"):
            self._seek_var.set(0.0)
        if self._sim_solutions_ready and self._sim_solutions:
            self._seek_to_fraction(0.0)

    def _seek_stop(self):
        """⏹ 停止（先頭へリセット）。"""
        self._seek_to_start()
        self.viewport.set_selected_waypoint(None)

    def _toggle_fast_mode(self):
        """軽量表示チェックボックス: 停止中・再生中いずれも即座に反映する。"""
        # 手動操作が入ったら自動切替の一時状態を解除し、手動設定を優先する。
        self._auto_fast_active = False
        self.viewport.set_fast_mode(self._fast_mode_var.get())

    def _pause_play(self):
        self._sim_playing = False
        self._sim_running = False
        self._play_btn_var.set("▶")
        if hasattr(self, "_sim_btn"):
            self._sim_btn.config(state="normal")

    # ── Simulation（互換: F5 / 実行ボタン → 先頭から再生） ──────────────

    def _start_simulation(self):
        """F5 / 実行ボタン: 先頭から再生する。"""
        # Route Sequence が定義されていれば、変更があれば展開して再生
        if self._route_sequence:
            needs_apply = self._seq_dirty or self.route.name != "ROUTE_SEQUENCE"
            if needs_apply:
                self._play_sequence()
                return
            # シーケンス適用済みかつ変更なし → 通常再生へ fall-through
        if not self.route.waypoints:
            messagebox.showwarning("経路点なし", "経路点が1つもありません。")
            return
        self._seek_var.set(0.0)
        self._play_from_current()

    def _play_from_current(self):
        """現在のシーク位置から前方へアニメーション再生する。"""
        if not self.route.waypoints:
            messagebox.showwarning("経路点なし", "経路点が1つもありません。")
            return
        if not self._sim_solutions_ready:
            self._set_status("⌛  IK事前計算中です。完了までお待ちください...")
            return
        if self._sim_playing:
            return
        if self._sim_thread and self._sim_thread.is_alive():
            return

        n = len(self.route.waypoints)
        if n < 2:
            return

        self._sim_playing = True
        self._sim_running = True
        self._play_btn_var.set("⏸")
        self._sim_btn.config(state="disabled")

        # 案2: 再生中は自動で軽量表示へ切替（手動でONでない場合のみ）。
        # 停止時に _playback_done で元の表示へ戻す。
        self._auto_fast_active = False
        if self._auto_fast_var.get() and not self._fast_mode_var.get():
            self._auto_fast_active = True
            self.viewport.set_fast_mode(True)

        cum, total = self._segment_times()
        start_idx = float(self._seek_var.get())
        # 末尾近くから再生開始した場合は先頭へ
        if start_idx >= n - 1 - 1e-6:
            start_idx = 0.0
            self._seek_var.set(0.0)

        # 開始位置に対応する経過時間
        def idx_to_time(idx_f):
            i0 = int(np.floor(idx_f))
            i1 = min(i0 + 1, n - 1)
            a = idx_f - i0
            return cum[i0] + a * (cum[i1] - cum[i0])

        def time_to_idx(t):
            return self._time_to_idx(t, cum, total, n)

        t_start = idx_to_time(start_idx)
        self._sim_start_time = time.time()
        self._sim_total_est = total
        self._sim_play_t0 = t_start

        def run():
            while self._sim_running:
                wall = time.time() - self._sim_start_time
                t = self._sim_play_t0 + wall
                idx_f = time_to_idx(t)

                def _update(idx=idx_f):
                    if not self._sim_running:
                        return
                    self._seek_updating = True
                    try:
                        self._seek_var.set(idx)
                    finally:
                        self._seek_updating = False
                    self._seek_to_fraction(idx)
                    pct = int(idx / max(n - 1, 1) * 100)
                    self._sim_progress_var.set(f"P[{int(idx)+1}]/{n}  {pct}%")
                self.root.after(0, _update)

                if t >= total:
                    break
                time.sleep(0.03)

            self.root.after(0, self._playback_done)

        self._sim_thread = threading.Thread(target=run, daemon=True)
        self._sim_thread.start()
        self._sim_tick()

    def _sim_tick(self):
        """再生時間ラベルを 0.1 秒ごとに更新する（フリーズ検知用）。"""
        if not self._sim_running:
            return
        wall = time.time() - self._sim_start_time
        elapsed = getattr(self, "_sim_play_t0", 0.0) + wall
        elapsed = min(elapsed, self._sim_total_est)
        self._sim_time_var.set(f"⏱  {elapsed:.1f}s / 約{self._sim_total_est:.1f}s")
        self.root.after(100, self._sim_tick)

    def _stop_simulation(self):
        """■ 停止（互換）: 再生を止め先頭へは戻さない。"""
        self._pause_play()

    def _playback_done(self):
        """再生が末尾に到達した／停止された後の後処理。"""
        was_running = self._sim_running
        self._sim_playing = False
        self._sim_running = False
        self._play_btn_var.set("▶")
        self._sim_btn.config(state="normal")
        # 案2: 自動軽量表示を解除し、手動設定の表示モードへ戻す
        if getattr(self, "_auto_fast_active", False):
            self._auto_fast_active = False
            self.viewport.set_fast_mode(self._fast_mode_var.get())
        n = len(self.route.waypoints)
        if was_running and n:
            # 末尾まで再生し切った場合は末尾に合わせる
            self._seek_updating = True
            try:
                self._seek_var.set(float(n - 1))
            finally:
                self._seek_updating = False
            self._seek_to_fraction(float(n - 1))
        elapsed = self._sim_total_est
        self._sim_progress_var.set("完了" if was_running else "一時停止")
        self._sim_time_var.set(f"⏱  完了 {elapsed:.1f}s")
        self._set_status(f"✔  シミュレーション完了（推定再生時間 {elapsed:.1f}s）")

    # ── 案1: 事前描画（プリレンダリング）再生 ──────────────────────────

    def _smooth_play(self):
        """全フレームを事前に画像へ描画してから滑らかに再生する。

        キャッシュ済みフレームがある場合は描画をスキップして即再生。
        ルート変更時は _invalidate_sim_solutions でキャッシュが自動破棄される。
        Route Sequence が定義されている場合は自動的に展開してから再生する。
        """
        # Route Sequence が定義されていれば、変更があれば展開してから滑らか再生
        if self._route_sequence:
            needs_apply = self._seq_dirty or self.route.name != "ROUTE_SEQUENCE"
            if needs_apply:
                flat_wps = self._flatten_sequence()
                if not flat_wps:
                    return
                self._push_undo("Route Sequence 展開（滑らか再生）")
                self.route.waypoints = flat_wps
                self.route.name = "ROUTE_SEQUENCE"
                self.route_editor.set_route(self.route)
                self._on_route_changed()
                self._seq_dirty = False
                self._set_status(
                    f"⌛  Route Sequence IK計算中 ({len(flat_wps)}点)…"
                    " 完了後に滑らか再生を開始します")
                token = self._sim_precompute_token
                def _wait_ik():
                    if self._sim_precompute_token != token:
                        return
                    if self._sim_solutions_ready:
                        self._smooth_play()  # 再帰: 今度は needs_apply=False で即再生
                    else:
                        self.root.after(200, _wait_ik)
                self.root.after(200, _wait_ik)
                return
            # シーケンス適用済みかつ変更なし → 通常の滑らか再生ロジックへ fall-through

        n_wp = len(self.route.waypoints)
        if n_wp < 2:
            messagebox.showwarning("経路点なし", "経路点が2つ以上必要です。")
            return
        if not self._sim_solutions_ready or not self._sim_solutions:
            self._set_status("⌛  IK事前計算中です。完了までお待ちください...")
            return
        if self._sim_playing or (self._sim_thread and self._sim_thread.is_alive()):
            return

        sols = self._sim_solutions
        ns = len(sols)
        cum, total = self._segment_times()
        if total <= 1e-6:
            messagebox.showinfo(
                "再生不可", "総再生時間が 0 秒です。経路点の速度を確認してください。")
            return

        # ── GPU(リアルタイム)バックエンドでは事前描画は不要 ──────────────
        # 実機メッシュのまま 60fps 超で再生できるため、リアルタイム再生へ委譲する。
        if getattr(self.viewport, "realtime", False):
            self._seek_var.set(0.0)
            self._play_from_current()
            return

        # ── キャッシュがあれば即再生 ──────────────────────────────────
        if self._pre_frames is not None and len(self._pre_frames) >= 2:
            self._smooth_start_playback(ns, total)
            return

        if not self._prerender_frames(ns, cum, total):
            return
        self._smooth_btn.config(text="🎬  滑らか再生（キャッシュ済み）")
        self._save_video_btn.config(state="normal")
        self._smooth_start_playback(ns, total)

    def _prerender_frames(self, ns: int, cum, total: float) -> bool:
        """全フレームを render_frame で画像化し _pre_frames/_pre_idxs に格納する。

        進捗ダイアログを表示し、キャンセル・失敗時は False を返す（キャッシュ未更新）。
        matplotlib 版の滑らか再生と GPU 版の動画保存の双方から使用する。
        """
        sols = self._sim_solutions

        # ── フレーム枚数（時間等間隔・メモリ上限でクランプ） ──
        fig = self.viewport.fig
        w = max(int(fig.get_figwidth() * fig.dpi), 1)
        h = max(int(fig.get_figheight() * fig.dpi), 1)
        frame_bytes = max(w * h * 4, 1)
        budget = 400 * 1024 * 1024
        fps = self._pre_fps
        n_frames = int(round(total * fps)) + 1
        n_frames = max(2, min(n_frames, 240, max(2, int(budget // frame_bytes))))
        times = np.linspace(0.0, total, n_frames)

        def time_to_idx(t):
            return self._time_to_idx(t, cum, total, ns)

        def idx_to_q(idx_f):
            idx_f = max(0.0, min(float(ns - 1), idx_f))
            i0 = int(np.floor(idx_f))
            i1 = min(i0 + 1, ns - 1)
            a = idx_f - i0
            return sols[i0] + a * (sols[i1] - sols[i0])

        idxs = [time_to_idx(t) for t in times]

        # ── 進捗ダイアログ（キャンセル可） ──
        dlg = tk.Toplevel(self.root)
        dlg.title("事前描画中…")
        dlg.transient(self.root)
        dlg.resizable(False, False)
        dlg.configure(bg=BG_PANEL)
        tk.Label(dlg, text="🎬 全フレームを描画しています…",
                 bg=BG_PANEL, fg=FG_PRIMARY,
                 font=("", 10, "bold")).pack(padx=16, pady=(14, 6))
        pbar = ttk.Progressbar(dlg, length=320, mode="determinate",
                               maximum=n_frames)
        pbar.pack(padx=16, pady=4)
        pct_var = tk.StringVar(value=f"0 / {n_frames}")
        tk.Label(dlg, textvariable=pct_var, bg=BG_PANEL, fg=FG_SUB,
                 font=("", 8)).pack(pady=(0, 4))
        cancel = {"flag": False}
        ttk.Button(dlg, text="キャンセル",
                   command=lambda: cancel.update(flag=True)).pack(pady=(0, 12))
        dlg.protocol("WM_DELETE_WINDOW", lambda: cancel.update(flag=True))
        dlg.update_idletasks()
        try:
            x = (self.root.winfo_rootx()
                 + (self.root.winfo_width() - dlg.winfo_width()) // 2)
            y = (self.root.winfo_rooty()
                 + (self.root.winfo_height() - dlg.winfo_height()) // 2)
            dlg.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        except Exception:
            pass
        dlg.grab_set()

        # ── 同期レンダリング（matplotlib はメインスレッド限定） ──
        frames = []
        saved_pose = self._joint_angles.copy()
        self._set_status("🎬  事前描画中…")
        for k, idx_f in enumerate(idxs):
            if cancel["flag"]:
                break
            try:
                frames.append(self.viewport.render_frame(idx_to_q(idx_f)))
            except Exception as e:
                dlg.grab_release()
                dlg.destroy()
                self.viewport.update_robot(saved_pose)
                messagebox.showerror("事前描画エラー", f"描画に失敗しました:\n{e}")
                self._set_status("✖  事前描画に失敗しました")
                return False
            pbar["value"] = k + 1
            pct_var.set(f"{k + 1} / {n_frames}")
            if (k % 2) == 0 or k == n_frames - 1:
                dlg.update()

        cancelled = cancel["flag"]
        dlg.grab_release()
        dlg.destroy()

        if cancelled or len(frames) < 2:
            self.viewport.update_robot(saved_pose)
            self._set_status("⏹  事前描画をキャンセルしました")
            return False

        n_frames = len(frames)
        idxs = idxs[:n_frames]

        # キャッシュとして保持（ルート変更まで繰り返し再生に使用）
        self._pre_frames = frames
        self._pre_idxs = idxs
        return True

    def _smooth_start_playback(self, ns: int, total: float):
        """キャッシュ済みフレームで再生ループを起動する。"""
        frames = self._pre_frames
        idxs = self._pre_idxs
        n_frames = len(frames)
        fps = self._pre_fps

        self.viewport.begin_prerendered_playback(frames[0])
        self._sim_playing = True
        self._sim_running = True
        self._play_btn_var.set("⏸")
        self._sim_btn.config(state="disabled")
        self._smooth_btn.config(state="disabled")
        self._save_video_btn.config(state="disabled")
        self._sim_start_time = time.time()
        self._sim_total_est = total
        self._sim_play_t0 = 0.0

        def run():
            while self._sim_running:
                wall = time.time() - self._sim_start_time
                fi = int(round(wall * fps))
                if fi >= n_frames:
                    fi = n_frames - 1

                def _update(fi=fi):
                    if not self._sim_running or self.viewport._pre_img is None:
                        return
                    self.viewport.show_prerendered_frame(frames[fi])
                    idx = idxs[fi]
                    self._seek_updating = True
                    try:
                        self._seek_var.set(idx)
                    finally:
                        self._seek_updating = False
                    pct = int(idx / max(ns - 1, 1) * 100)
                    self._sim_progress_var.set(f"P[{int(idx)+1}]/{ns}  {pct}%")
                self.root.after(0, _update)

                if wall >= total:
                    break
                time.sleep(1.0 / fps)
            self.root.after(0, self._smooth_playback_done)

        self._sim_thread = threading.Thread(target=run, daemon=True)
        self._sim_thread.start()
        self._sim_tick()

    def _smooth_playback_done(self):
        """事前描画再生が末尾に到達した／停止された後の後処理。"""
        was_running = self._sim_running
        self._sim_playing = False
        self._sim_running = False
        self._play_btn_var.set("▶")
        self._sim_btn.config(state="normal")
        # キャッシュが残っていれば「キャッシュ済み」表示のまま、動画保存も有効
        cached = self._pre_frames is not None and len(self._pre_frames) >= 2
        self._smooth_btn.config(
            state="normal",
            text="🎬  滑らか再生（キャッシュ済み）" if cached
                 else "🎬  滑らか再生（事前描画）")
        self._save_video_btn.config(state="normal" if cached else "disabled")
        n_done = len(self._pre_frames) if self._pre_frames else 0
        self.viewport.end_prerendered_playback()
        n = len(self.route.waypoints)
        if was_running and n:
            self._seek_updating = True
            try:
                self._seek_var.set(float(n - 1))
            finally:
                self._seek_updating = False
            self._seek_to_fraction(float(n - 1))
        elapsed = self._sim_total_est
        self._sim_progress_var.set("完了" if was_running else "一時停止")
        self._sim_time_var.set(f"⏱  完了 {elapsed:.1f}s")
        self._set_status(
            f"✔  滑らか再生完了（{n_done} フレーム キャッシュ済み・繰り返し再生可）")

    def _save_smooth_video(self):
        """キャッシュ済みフレームを動画ファイル（MP4 / GIF）として保存する。

        GPU(リアルタイム)バックエンドでは滑らか再生で事前描画を行わないため、
        フレーム未キャッシュ時はここで動画用に一度だけ事前描画する。
        """
        if (not self._pre_frames or len(self._pre_frames) < 2) \
                and getattr(self.viewport, "realtime", False):
            if not self._sim_solutions_ready or len(self._sim_solutions) < 2:
                messagebox.showwarning("フレームなし",
                                       "経路点が2つ以上必要です（IK計算の完了もお待ちください）。")
                return
            cum, total = self._segment_times()
            if total <= 1e-6:
                messagebox.showinfo("保存不可", "総再生時間が 0 秒です。")
                return
            if not self._prerender_frames(len(self._sim_solutions), cum, total):
                return   # キャンセル/失敗

        if not self._pre_frames or len(self._pre_frames) < 2:
            messagebox.showwarning("フレームなし",
                                   "先に「滑らか再生」を実行してフレームをキャッシュしてください。")
            return

        path = filedialog.asksaveasfilename(
            title="動画を保存",
            defaultextension=".mp4",
            filetypes=[("MP4 動画", "*.mp4"), ("GIF アニメーション", "*.gif"),
                       ("すべてのファイル", "*.*")],
        )
        if not path:
            return

        fps = self._pre_fps
        ext = os.path.splitext(path)[1].lower()
        self._set_status(f"💾  動画を保存中…  {os.path.basename(path)}")
        self._save_video_btn.config(state="disabled")
        self.root.update()

        try:
            if ext == ".gif":
                self._save_as_gif(path, fps)
            else:
                # imageio の有無を先に確認し、なければ GIF フォールバックを提案
                try:
                    import imageio as _imageio_check  # noqa: F401
                except ImportError:
                    gif_path = os.path.splitext(path)[0] + ".gif"
                    if messagebox.askyesno(
                        "imageio が見つかりません",
                        "MP4 保存には imageio[ffmpeg] が必要です。\n"
                        "インストールするには:\n"
                        "  pip install imageio[ffmpeg]\n\n"
                        f"代わりに GIF として保存しますか？\n{gif_path}",
                    ):
                        path = gif_path
                        self._save_as_gif(path, fps)
                        self._set_status(f"✔  GIF を保存しました: {path}")
                    else:
                        self._set_status("動画の保存をキャンセルしました")
                    return
                self._save_as_mp4(path, fps)
            self._set_status(f"✔  動画を保存しました: {path}")
        except Exception as e:
            # MP4 書き込みに失敗した場合は GIF フォールバックを提案する
            if ext != ".gif":
                gif_path = os.path.splitext(path)[0] + ".gif"
                if messagebox.askyesno(
                    "MP4 保存に失敗",
                    f"MP4 の保存に失敗しました:\n{e}\n\n"
                    f"代わりに GIF として保存しますか？\n{gif_path}",
                ):
                    try:
                        self._save_as_gif(gif_path, fps)
                        self._set_status(f"✔  GIF を保存しました: {gif_path}")
                        return
                    except Exception as e2:
                        messagebox.showerror(
                            "保存エラー", f"GIF の保存にも失敗しました:\n{e2}")
                        self._set_status("✖  動画の保存に失敗しました")
                        return
            messagebox.showerror("保存エラー", f"動画の保存に失敗しました:\n{e}")
            self._set_status("✖  動画の保存に失敗しました")
        finally:
            self._save_video_btn.config(state="normal")

    def _save_as_mp4(self, path: str, fps: float):
        """imageio-ffmpeg で MP4 として保存する。"""
        try:
            import imageio
        except ImportError:
            raise RuntimeError(
                "imageio が見つかりません。pip install imageio[ffmpeg] を実行してください。")
        # RGBA→RGB（alpha チャンネルを落とす）
        rgb_frames = [f[:, :, :3] for f in self._pre_frames]
        # libx264 / yuv420p は偶数の幅・高さを要求する。奇数サイズ
        # （例: 1223x858）だと ffmpeg が Broken pipe で即死するため、
        # 末尾の 1 行 / 1 列を削って偶数サイズにそろえてから書き出す。
        h, w = rgb_frames[0].shape[:2]
        h2, w2 = h - (h % 2), w - (w % 2)
        if (h2, w2) != (h, w):
            rgb_frames = [np.ascontiguousarray(f[:h2, :w2, :])
                          for f in rgb_frames]
        # macro_block_size=1: 偶数化済みなので追加パディング不要
        imageio.mimsave(path, rgb_frames, fps=fps,
                        codec="libx264", quality=8,
                        macro_block_size=1)

    def _save_as_gif(self, path: str, fps: float):
        """Pillow で GIF アニメーションとして保存する。"""
        from PIL import Image
        duration_ms = int(1000 / max(fps, 1))
        pil_frames = [Image.fromarray(f[:, :, :3]) for f in self._pre_frames]
        pil_frames[0].save(
            path, save_all=True, append_images=pil_frames[1:],
            duration=duration_ms, loop=0, optimize=False)

    # ──────────────────────────────────────────────────────────────────
    # IK
    # ──────────────────────────────────────────────────────────────────

    def _compute_ik_for_wp(self):
        idx = self._ik_wp_var.get() - 1
        if idx < 0 or idx >= len(self.route.waypoints):
            messagebox.showwarning("範囲外", f"P[{idx+1}] は存在しません。")
            return
        wp = self.route.waypoints[idx]
        T  = wp.to_transform()
        self._set_status(f"IK 計算中: P[{idx+1}] ({wp.label})...")
        self.root.update()
        q, ok = self.kin.inverse(T, q_init=self._joint_angles)
        if ok:
            self._push_undo("IK移動", coalesce=True)
            self._set_angles(q)
            self.viewport.set_selected_waypoint(idx)
            self._set_status(f"✔  IK 成功: P[{idx+1}] ({wp.label})")
        else:
            messagebox.showwarning("IK 失敗",
                f"P[{idx+1}] ({wp.label}) の逆運動学計算に失敗しました。\n"
                f"位置が可動範囲外の可能性があります。\n"
                f"  X={wp.x:.1f}, Y={wp.y:.1f}, Z={wp.z:.1f} mm")

    # ──────────────────────────────────────────────────────────────────
    # Robot presets
    # ──────────────────────────────────────────────────────────────────

    def _go_home(self):
        self._push_undo("ホーム移動")
        self._set_angles(self.kin.dh.home_position())
        self._set_status("✔  ホームポジション (全軸 0°) に移動しました")

    def _go_ready(self):
        self._push_undo("レディ移動")
        self._set_angles(self.kin.dh.ready_position())
        self._set_status("✔  レディポジション (J2=-45° J3=+30° J5=-60°) に移動しました")

    def _set_angles(self, q: np.ndarray):
        self._joint_angles = q.copy()
        # スライダー var.set はコールバック (_on_slider_change) を誘発する
        # ため、プログラム由来の更新は undo に記録しない
        prev = self._suppress_undo
        self._suppress_undo = True
        try:
            for i, var in enumerate(self._slider_vars):
                var.set(np.rad2deg(q[i]))
        finally:
            self._suppress_undo = prev
        self._update_viewport_from_angles(q)
        self._update_fk_display()

    # ──────────────────────────────────────────────────────────────────
    # Tool / User frame dialogs
    # ──────────────────────────────────────────────────────────────────

    def _edit_tool_frame(self, tf=None):
        """ツールフレーム編集（tf 省略時はアクティブツール）。"""
        tf = tf if tf is not None else self._active_tool

        def _apply():
            if tf is self._active_tool:
                self.viewport.set_tool_frame(tf)
            if hasattr(self, "_tree"):
                self._tree_refresh()

        self._frame_editor_dialog(
            title=f"ツールフレーム編集: UT{tf.number} {tf.name}",
            desc="フランジ（J6先端）からTCP（ツール中心点）までのオフセットを設定します。\n包丁の場合、刃の中心まで Z方向に延長します。",
            obj=tf,
            fields=["x", "y", "z", "rx", "ry", "rz"],
            labels=["X オフセット (mm)", "Y オフセット (mm)", "Z オフセット (mm)",
                    "Rx 回転 (°)", "Ry 回転 (°)", "Rz 回転 (°)"],
            on_apply=_apply,
            undo_label="ツールフレーム編集",
        )

    def _edit_user_frame(self, uf=None):
        """ユーザーフレーム編集（uf 省略時はアクティブ UFrame）。

        UF9 STONE の場合は _sync_uf9 で参照フレーム・調整パネル・
        コンボボックスまで一括同期する（kenma 生成座標と一致を維持）。
        """
        uf = uf if uf is not None else self._active_uframe

        def _apply():
            if uf.number == 9:
                self._sync_uf9(uf)
            else:
                if uf is self._active_uframe:
                    self.viewport.set_user_frame(uf)
                if hasattr(self, "_tree"):
                    self._tree_refresh()

        self._frame_editor_dialog(
            title=f"ユーザーフレーム編集: UF{uf.number} {uf.name}",
            desc="作業座標系の原点を設定します。\n砥石の場合、砥石面の中心をユーザーフレーム原点とします。",
            obj=uf,
            fields=["x", "y", "z", "rx", "ry", "rz"],
            labels=["X 位置 (mm)", "Y 位置 (mm)", "Z 位置 (mm)",
                    "Rx 回転 (°)", "Ry 回転 (°)", "Rz 回転 (°)"],
            on_apply=_apply,
            undo_label="ユーザーフレーム編集",
        )

    def _frame_editor_dialog(self, title, desc, obj, fields, labels, on_apply,
                             undo_label="フレーム編集", fmt="{:.2f}"):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("380x340")
        win.configure(bg=BG_DARK)
        win.resizable(False, False)

        tk.Label(win, text=title, bg=BG_DARK, fg=ACCENT,
                 font=self._fnt(10, "bold")).pack(pady=(12, 2), padx=12, anchor="w")
        tk.Label(win, text=desc, bg=BG_DARK, fg=FG_SUB,
                 font=self._fnt(8), justify="left",
                 wraplength=350).pack(padx=12, anchor="w")

        ttk.Separator(win).pack(fill=tk.X, padx=12, pady=8)

        vars_ = {}

        def _live_apply():
            """各フィールドの値を obj に書き戻し 3D ビューへ即時反映する。"""
            self._push_undo(undo_label, coalesce=True)
            for f, v in vars_.items():
                try:
                    setattr(obj, f, float(v.get()))
                except (ValueError, tk.TclError):
                    pass
            on_apply()

        for f, lbl in zip(fields, labels):
            row = ttk.Frame(win)
            row.pack(fill=tk.X, padx=16, pady=2)
            tk.Label(row, text=lbl, bg=BG_PANEL, fg=FG_SUB,
                     font=self._fnt(8, fam=""), width=18, anchor="w").pack(side=tk.LEFT)
            raw = getattr(obj, f)
            v = tk.StringVar(value=fmt.format(float(raw)))
            vars_[f] = v
            # ホイール/Enter/フォーカス離脱で即時プレビュー
            self._make_num_field(row, v, on_change=_live_apply,
                                 width=12, fmt=fmt).pack(side=tk.LEFT, padx=4)

        def apply():
            _live_apply()
            self._set_status(f"✔  {title} を更新しました")
            win.destroy()

        btn_row = ttk.Frame(win)
        btn_row.pack(pady=12)
        ttk.Button(btn_row, text="適用して閉じる",
                   style="Primary.TButton", command=apply).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_row, text="キャンセル", command=win.destroy).pack(side=tk.LEFT, padx=6)

    # ──────────────────────────────────────────────────────────────────
    # Route operations
    # ──────────────────────────────────────────────────────────────────

    def _load_kenma_route(self):
        """研磨経路CSV（kenma形式）を読み込む。"""
        assets = _asset_path()
        default = os.path.join(assets, "kenma_route.csv")
        path = filedialog.askopenfilename(
            title="研磨経路CSV を開く（kenma形式）",
            initialdir=assets if os.path.exists(assets) else ".",
            initialfile="kenma_route.csv",
            filetypes=[("Kenma route CSV", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            loaded = RouteCSVIO.route_from_csv(path)
            if not loaded.waypoints:
                messagebox.showwarning("経路点なし",
                    "CSVに有効な経路点が見つかりませんでした。\n"
                    "ヘッダー行: x_mm,y_mm,z_mm,rx_deg,ry_deg,rz_deg,speed_mmps,motion_type,label")
                return
            self._push_undo("研磨経路CSV読込")
            self.route.waypoints = loaded.waypoints
            self.route.name      = loaded.name or os.path.splitext(os.path.basename(path))[0]
            self.route.comment   = loaded.comment or "Knife sharpening route"
            if loaded.uframe:
                self.route.uframe = loaded.uframe
            if loaded.utool:
                self.route.utool = loaded.utool
            self.route_editor.set_route(self.route)
            self.viewport.set_route(self.route)   # set_route が再描画する
            self._invalidate_sim_solutions()
            self._set_status(
                f"✔  研磨経路読込完了: {len(self.route)} 点 "
                f"(UF{self.route.uframe}/UT{self.route.utool}) ← {os.path.basename(path)}"
            )
        except Exception as e:
            messagebox.showerror("読込エラー", f"研磨経路CSV の読込に失敗しました:\n{e}")

    # 研磨機 STL の既定配置パラメータ（ユーザー確認済みの値）
    _STL_DEFAULT_POSE = (740.0, 240.0, 266.0, 0.0, 0.0, -90.0)

    def _apply_stl_default_pose(self):
        """STL既定位置をビューポートおよび入力欄に反映する。"""
        ix, iy, iz, irx, iry, irz = self._STL_DEFAULT_POSE
        self.viewport.set_stl_pose(ix, iy, iz, irx, iry, irz)
        if len(self._stl_pose_vars) >= 6:
            for var, val in zip(self._stl_pose_vars,
                                [ix, iy, iz, irx, iry, irz]):
                var.set(f"{val:.2f}")

    def _load_tormek_stl(self):
        """Tormek T8 STL のみ読み込む（研削経路CSVは読み込まない）。"""
        stl_path = os.path.realpath(_asset_path("Tormek_T8.stl"))
        if not os.path.isfile(stl_path):
            self._set_status("⚠  STL ファイルが見つかりません: " + stl_path)
            return
        with self._undo_group("Tormek STL 読込"):
            ok = self.viewport.load_stl(stl_path)
            if ok:
                self._apply_stl_default_pose()
                # 砥石を床（Z=0）に置く: 下面が Z=0 になるよう pose Z を自動調整
                stone_bottom_z = self._stl_bottom_z(0.0)
                ix, iy, iz, irx, iry, irz = self._STL_DEFAULT_POSE
                adjusted_z = iz - stone_bottom_z  # bottom=0 となる Z
                self.viewport.set_stl_pose(ix, iy, adjusted_z, irx, iry, irz)
                for var, val in zip(self._stl_pose_vars,
                                    [ix, iy, adjusted_z, irx, iry, irz]):
                    var.set(f"{val:.2f}")
                # UF9 STONE を STL 上面 Z で同期（USER_FRAMES / コンボ /
                # 参照フレーム / 調整パネルを一括更新）
                stone_top_z = self._stl_top_z(UserFrame.stone9().z)
                self._sync_uf9(self._uf9_frame(z=stone_top_z))
                self._set_status(
                    f"✔  Tormek T8 STL 読込済（底面=床 Z=0, 上面=UF9 z={stone_top_z:.0f}mm, Rz=-90°）")
            else:
                self._set_status("⚠  STL 読込失敗")

    def _load_tormek_csv(self):
        """Tormek 研削経路 CSV のみ読み込む（STLは触らない）。"""
        csv_path = _asset_path("grinding_path_sample.csv")
        if not os.path.exists(csv_path):
            self._set_status("⚠  研削経路 CSV が見つかりません: " + csv_path)
            return
        self._push_undo("Tormek CSV 読込")
        ok = self.viewport.load_csv_points(csv_path)
        if ok:
            self._set_status("✔  Tormek 研削経路 CSV 読込済")
        else:
            self._set_status("⚠  CSV 読込失敗")

    # ──────────────────────────────────────────────────────────────────
    # kenma形式 3ファイルLS出力（HaL / HaR / kenma）
    # ──────────────────────────────────────────────────────────────────

    def _kenma_inputs(self):
        """kenma生成入力 (blade_pts, blade_normals, T_blade, T_contact, csv名)。

        刃先CSVは読み込み済みのものを優先し、なければ同梱サンプルを使用。
        UF9 STONE は USER_FRAMES → ビューポート参照フレーム → 既定値の順。
        """
        if self.viewport.has_blade():
            pts = self.viewport._blade_pts
            nrm = self.viewport._blade_normals
            csv_name = os.path.basename(self._blade_csv_path or "blade.csv")
        else:
            sample = os.path.realpath(_asset_path("blade_sample.csv"))
            if not os.path.isfile(sample):
                raise FileNotFoundError(
                    "刃先CSVが読み込まれておらず、assets/blade_sample.csv も見つかりません")
            pts, nrm = load_blade_csv_file(sample)
            csv_name = os.path.basename(sample)

        vals = [float(v.get()) for v in self._blade_pose_vars]
        T_blade = Kinematics.pose_to_transform(*vals)

        T_contact = None
        for uf in self.USER_FRAMES:
            if uf.number == 9:
                T_contact = uf.to_transform()
                break
        if T_contact is None:
            for f in self.viewport.get_ref_frames():
                if f["name"] == "UF9: STONE":
                    T_contact = f["T"]
                    break
        if T_contact is None:
            # 既定の UF9 STONE（_setup_stone_uframe / stone9 と同じ値）
            T_contact = UserFrame.stone9().to_transform()
        return pts, nrm, T_blade, T_contact, csv_name

    def _generate_kenma(self):
        """kenma形式プログラム一式を生成して返す（IK計算あり・時間がかかる）。"""
        pts, nrm, T_blade, T_contact, csv_name = self._kenma_inputs()
        self._set_status(
            "kenma形式ルート生成中 — IK計算しています（数十秒かかる場合があります）...")
        self.root.update()
        result = generate_kenma_programs(pts, nrm, T_blade, T_contact, self.kin)
        return result, csv_name

    def _export_kenma_ls(self):
        """kenma形式LS（kenma.LS / HaL.LS / HaR.LS の3ファイル1組）を出力する。"""
        out_dir = filedialog.askdirectory(
            title="kenma形式LS（3ファイル1組）の出力先フォルダを選択")
        if not out_dir:
            return
        try:
            result, csv_name = self._generate_kenma()
            paths = export_kenma_ls(result, out_dir, self.kin,
                                    mn_comments=[csv_name])
        except Exception as e:
            messagebox.showerror("kenma形式LS出力エラー",
                                 f"出力に失敗しました:\n{e}")
            self._set_status("⚠  kenma形式LS出力に失敗しました")
            return

        names = " / ".join(os.path.basename(p) for p in paths)
        self._set_status(
            f"✔  kenma形式LS出力完了: {names}  "
            f"(到達不能 {result.n_unreachable}点) → {out_dir}")

        msg = (
            "kenma形式LSを出力しました。\n\n"
            f"出力先: {out_dir}\n"
            + "\n".join(f"  ・{os.path.basename(p)}" for p in paths)
            + f"\n\n左 {result.n_groups_left} / 右 {result.n_groups_right} グループ"
            "（各グループ: ホバー→接触→ストローク5点→リトラクト）\n"
            f"IK到達不能: {result.n_unreachable} 点"
        )
        if result.unreachable_labels:
            shown = ", ".join(result.unreachable_labels[:8])
            if len(result.unreachable_labels) > 8:
                shown += " ..."
            msg += f"\n（{shown}）"
        msg += ("\n\n生成した全シーケンスを経路点リストへ展開しますか？\n"
                "（シミュレーション実行で動作を再生できます）")
        if messagebox.askyesno("kenma形式LS出力", msg):
            self._apply_kenma_sequence(result)

    def _apply_kenma_sequence(self, result):
        """生成済み kenma シーケンス（CALL展開済み）を経路点リストへ反映する。

        一括反映: リスト構築 → set_route 1回 → 再描画 1回（per-point 処理なし）。
        """
        self._push_undo("研磨ルート展開")
        seq = build_playback_sequence(result)
        self.route.waypoints = seq
        self.route.name = "kenma"
        self.route.comment = "kenma 3-file sequence"
        self.route.uframe = 9
        self.route.utool = 9
        self.route_editor.set_route(self.route)   # リスト一括再構築（1回）
        self.viewport.set_route(self.route)       # 再描画（1回・軽量モード）
        if hasattr(self, "_tree"):
            self._tree_refresh()
        self._invalidate_sim_solutions()
        self._set_status(
            f"✔  kenma形式ルートを経路点リストへ展開: {len(seq)} 点 "
            f"(到達不能 {result.n_unreachable}点) — シミュ実行で再生できます")

    # ──────────────────────────────────────────────────────────────────
    # RoboDK風 曲線選択 → 研磨ルート生成
    # ──────────────────────────────────────────────────────────────────

    def _kenma_curve_select_dialog(self):
        """曲線を選択して研磨ルート生成（RoboDK の Curve Follow 曲線選択相当）。

        刃先CSVのストロークグループ（=「角で曲線を分ける」結果に相当）を
        3Dビューポートにクリック可能な曲線として表示し、クリックした順番が
        研磨の実行順になる。左側面の選択は HaL、右側面は HaR に振り分けられる。

        クリック判定: ドラッグ距離 5px 未満のリリースをクリックとして扱う。
        ドラッグ（5px 以上）は従来どおり視点回転 — 選択モード切替は不要。
        """
        from ..path.kenma_export import detect_stroke_groups

        try:
            pts, nrm, T_blade, T_contact, csv_name = self._kenma_inputs()
            groups = detect_stroke_groups(pts, nrm)
        except Exception as e:
            messagebox.showerror("曲線選択",
                                 f"刃先CSVの曲線検出に失敗しました:\n{e}")
            return

        # 命名: 側（左=L/右=R）+ 刃渡り位置順（CSV順 = 刃元→刃先）
        names: list = []
        sides: list = []
        n_left = n_right = 0
        for g in groups:
            if float(np.mean(g.normals[:, 0])) < 0.0:
                n_left += 1
                names.append(f"L{n_left:02d}")
                sides.append("L")
            else:
                n_right += 1
                names.append(f"R{n_right:02d}")
                sides.append("R")

        # 曲線は刃先CSVローカル座標のまま渡す。ビューポートが描画のたびに
        # 現在のフランジ姿勢 (T_ee @ _blade_T) でワールドへ変換するため、
        # ジョグ・シミュレーション再生中も包丁（刃先オーバーレイ）に追従する。
        local_curves = [g.pts for g in groups]

        # 前回のダイアログ状態（選択順・逆方向・オフセット式）を復元する。
        # 曲線名で照合するため、同じ刃先CSVなら選択がそのまま戻る。
        saved = self._kenma_dlg_state
        name_to_gi = {n: i for i, n in enumerate(names)}
        sel: list = [name_to_gi[n] for n in saved.get("sel", [])
                     if n in name_to_gi]          # 選択順のグループ index 列
        rev: dict = {name_to_gi[n]: True          # gi -> 逆方向フラグ
                     for n, r in saved.get("rev", {}).items()
                     if r and n in name_to_gi}

        def _save_dlg_state():
            """選択順・逆方向・オフセット式を保持する（次回・undo 用）。"""
            self._kenma_dlg_state = {
                "offset": off_var.get(),
                "sel": [names[gi] for gi in sel],
                "rev": {names[gi]: True for gi in sel if rev.get(gi, False)},
            }

        win = tk.Toplevel(self.root)
        win.title("曲線を選択して研磨ルート生成")
        win.geometry("470x760")
        win.configure(bg=BG_DARK)
        win.resizable(False, False)

        tk.Label(win, text="📐  曲線を選択して研磨ルート生成",
                 bg=BG_DARK, fg=ACCENT,
                 font=self._fnt(12, "bold")).pack(pady=(12, 2), padx=14, anchor="w")
        tk.Label(win,
            text="3Dビューの水色の曲線をクリックすると選択（緑・番号付き）されます。\n"
                 "クリックした順番 = 研磨の実行順（RoboDKの曲線選択と同じ）。\n"
                 "選択済み曲線を再クリックすると解除されます。\n"
                 "※ クリック（5px未満）で選択 / ドラッグはそのまま視点回転です。\n"
                 "※ リストの右クリック or「⇄ 逆方向」で掃引方向を反転できます。\n"
                 "※ 生成後もこの画面は開いたまま — オフセット等を変えて何度でも\n"
                 "　 生成し直せます（Ctrl+Z で生成前のルートに戻せます）。",
            bg=BG_DARK, fg=FG_SUB,
            font=self._fnt(8), justify="left").pack(padx=14, anchor="w")

        ttk.Separator(win).pack(fill=tk.X, padx=14, pady=6)

        # 選択リスト（実行順）
        lb_frame = ttk.Frame(win)
        lb_frame.pack(fill=tk.BOTH, expand=True, padx=14)
        sb = tk.Scrollbar(lb_frame, orient=tk.VERTICAL)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        lb = tk.Listbox(
            lb_frame, height=10, yscrollcommand=sb.set,
            bg=BG_WIDGET, fg=FG_PRIMARY, font=("Consolas", 9),
            selectbackground=BTN_PRIMARY, selectforeground="white",
            borderwidth=0, highlightthickness=1, highlightcolor=BORDER,
            activestyle="none")
        lb.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        sb.config(command=lb.yview)
        _tip(lb, "→ = 順方向（CSV順: 刃元→刃先）\n"
                 "← = 逆方向（接触点順を逆転して掃引）\n"
                 "右クリック or「⇄ 逆方向」ボタンで切替できます")

        info_var = tk.StringVar()
        tk.Label(win, textvariable=info_var, bg=BG_DARK, fg=ACCENT2,
                 font=self._fnt(9, "bold", fam="")).pack(padx=14, pady=(4, 0), anchor="w")

        # スタート関節候補の状態（選択内容が変わったら再計算が必要）
        q_branches: list = []          # List[np.ndarray]
        q_state = {"key": None}        # 候補計算時の (先頭gi, rev) キー

        def _first_key():
            return (sel[0], bool(rev.get(sel[0], False))) if sel else None

        def _invalidate_branches():
            """先頭曲線が変わったら関節候補を無効化する。"""
            if q_state["key"] is not None and q_state["key"] != _first_key():
                q_branches.clear()
                q_state["key"] = None
                q_combo.config(values=[])
                q_var.set("")
                q_info_var.set("先頭曲線が変わりました — 「候補を計算」を再実行してください")

        def _refresh(keep_sel: Optional[int] = None):
            orders = [None] * len(groups)
            for k, gi in enumerate(sel):
                orders[gi] = k + 1
            self.viewport.set_pick_orders(orders)
            lb.delete(0, tk.END)
            for k, gi in enumerate(sel):
                g = groups[gi]
                y0 = float(g.pts[:, 1].min())
                y1 = float(g.pts[:, 1].max())
                arrow = "←" if rev.get(gi, False) else "→"
                lb.insert(tk.END,
                          f"{k+1:2d}. {names[gi]} {arrow}  刃渡り {y0:6.1f}〜{y1:6.1f}mm")
            if keep_sel is not None and 0 <= keep_sel < lb.size():
                lb.selection_set(keep_sel)
                lb.see(keep_sel)
            nl = sum(1 for gi in sel if sides[gi] == "L")
            nr = len(sel) - nl
            info_var.set(
                f"選択 {len(sel)} / 全{len(groups)} 曲線"
                f"（HaL: {nl} / HaR: {nr}・残り {len(groups) - len(sel)}）")
            _invalidate_branches()

        def _on_pick(gi: int):
            if gi in sel:
                sel.remove(gi)            # 再クリックで解除（以降を再採番）
                rev.pop(gi, None)
                self._set_status(f"曲線 {names[gi]} の選択を解除しました")
            else:
                sel.append(gi)            # クリック順に追加
                self._set_status(
                    f"曲線 {names[gi]} を選択（{len(sel)}番目）")
            _refresh()

        def _select_all():
            sel.clear()
            rev.clear()
            sel.extend(range(len(groups)))   # 標準順 = CSV順（従来の全曲線生成と同順）
            _refresh()

        def _reset():
            sel.clear()
            rev.clear()
            _refresh()

        def _move(delta: int):
            cur = lb.curselection()
            if not cur:
                return
            i = cur[0]
            j = i + delta
            if 0 <= j < len(sel):
                sel[i], sel[j] = sel[j], sel[i]
                _refresh(keep_sel=j)

        def _toggle_reverse(row: Optional[int] = None):
            """ハイライト行（または指定行）の掃引方向を反転する。"""
            if row is None:
                cur = lb.curselection()
                if not cur:
                    self._set_status("⚠  逆方向にする曲線をリストから選択してください")
                    return
                row = cur[0]
            if not (0 <= row < len(sel)):
                return
            gi = sel[row]
            rev[gi] = not rev.get(gi, False)
            self._set_status(
                f"曲線 {names[gi]} を{'逆方向 ←' if rev[gi] else '順方向 →'}に設定しました")
            _refresh(keep_sel=row)

        def _on_lb_right_click(event):
            row = lb.nearest(event.y)
            if 0 <= row < len(sel):
                lb.selection_clear(0, tk.END)
                lb.selection_set(row)
                _toggle_reverse(row)

        lb.bind("<Button-3>", _on_lb_right_click)

        def _close():
            _save_dlg_state()
            self.viewport.clear_pick_curves()
            win.destroy()

        # ── パスからツールへのオフセット（RoboDK 互換の式入力） ──────
        off_lf = ttk.LabelFrame(win, text="  パスからツールへのオフセット")
        off_lf.pack(fill=tk.X, padx=14, pady=(6, 2))
        _tip(off_lf,
             "パス（砥石接触フレーム）に対するツールの追加オフセット式です。\n"
             "transl(x,y,z) [mm] / rotx(°) / roty(°) / rotz(°) を * で連結します。\n\n"
             "回転は接触点を固定したまま姿勢のみ変化します:\n"
             "・rotx = 砥石接線（刃渡り方向）まわり ＝ 刃付け角度の軸\n"
             "・roty = 送り方向まわり（刃の前後倒れ）\n"
             "・rotz = 砥石面法線まわり（面内の向き）\n"
             "transl は接触フレーム軸方向に接触点を平行移動します。\n\n"
             "例: rotx(15) → 刃付け角を15°追加\n"
             "既定 rotx(0)*roty(0)*rotz(0) = オフセットなし（従来と同一出力）\n"
             "※ 式は保持されます — 生成後に変更して何度でも生成し直せます")
        off_var = tk.StringVar(
            value=saved.get("offset", "rotx(0)*roty(0)*rotz(0)"))
        off_state = {"mat": np.eye(4)}   # 最後に有効だった行列を保持
        off_row = ttk.Frame(off_lf)
        off_row.pack(fill=tk.X, padx=6, pady=(2, 1))
        off_entry = tk.Entry(off_row, textvariable=off_var,
                             bg=BG_WIDGET, fg=FG_PRIMARY,
                             insertbackground=FG_PRIMARY,
                             relief="flat", highlightthickness=1,
                             highlightbackground=BORDER,
                             font=("Consolas", 9))
        off_entry.pack(fill=tk.X, expand=True)
        off_msg_var = tk.StringVar(value="")
        tk.Label(off_lf, textvariable=off_msg_var, bg=BG_PANEL, fg=ERR_RED,
                 font=self._fnt(7, fam=""), anchor="w").pack(
                     fill=tk.X, padx=6, pady=(0, 3))

        def _validate_offset(event=None) -> bool:
            """式を検証し、成功なら off_state['mat'] を更新する。"""
            try:
                off_state["mat"] = parse_pose_expression(off_var.get())
                off_entry.config(bg=BG_WIDGET, highlightbackground=BORDER)
                off_msg_var.set("")
                return True
            except ValueError as e:
                off_entry.config(bg="#4A1E1E", highlightbackground=ERR_RED)
                off_msg_var.set(f"⚠ 式エラー: {e}（直前の有効な式を使用します）")
                self._set_status(f"⚠  オフセット式エラー: {e}")
                return False

        off_entry.bind("<KeyRelease>", _validate_offset)
        off_entry.bind("<FocusOut>",  _validate_offset)
        _validate_offset()   # 復元した式を即評価（off_state["mat"] を同期）

        # ── スタート地点に好ましい関節（RoboDK 互換） ────────────────
        q_lf = ttk.LabelFrame(win, text="  スタート地点に好ましい関節")
        q_lf.pack(fill=tk.X, padx=14, pady=(2, 2))
        _tip(q_lf,
             "先頭に選択した曲線のホバー姿勢に対する IK 解候補（関節配置の\n"
             "ブランチ）を列挙します（RoboDK の同名機能と同等）。\n"
             "「候補を計算」→ J1〜J6 [deg] の候補から開始関節を選択します。\n"
             "選択した関節はルート生成全体の IK シードとなり、そのまま\n"
             "P[1]/HOME の関節姿勢として LS に出力されます。\n"
             "未計算・未選択の場合は従来どおり自動選択（レディ姿勢に最近傍）です。")
        q_row = ttk.Frame(q_lf)
        q_row.pack(fill=tk.X, padx=6, pady=(3, 1))
        q_var = tk.StringVar(value="")
        q_combo = ttk.Combobox(q_row, textvariable=q_var, values=[],
                               state="readonly", width=34,
                               font=("Consolas", 8))
        q_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        _tip(q_combo, "J1, J2, J3, J4, J5, J6 [deg] — 上ほどレディ姿勢に近い候補です")

        q_info_var = tk.StringVar(value="未計算（自動選択）")
        tk.Label(q_lf, textvariable=q_info_var, bg=BG_PANEL, fg=FG_SUB,
                 font=self._fnt(7, fam=""), anchor="w").pack(
                     fill=tk.X, padx=6, pady=(0, 3))

        def _compute_branches():
            if not sel:
                messagebox.showwarning("曲線未選択",
                    "先に曲線を選択してください。", parent=win)
                return
            if not _validate_offset():
                pass  # 直前の有効な式で続行
            self._set_status("スタート関節候補を計算中（IK ブランチ列挙）...")
            q_info_var.set("計算中...")
            win.update()
            try:
                order = [(gi, bool(rev.get(gi, False))) for gi in sel]
                T_hover = first_hover_T(
                    pts, nrm, T_blade, T_contact,
                    selected_groups=order,
                    tool_offset=off_state["mat"])
                branches = enumerate_ik_branches(self.kin, T_hover)
            except Exception as e:
                messagebox.showerror("候補計算エラー",
                                     f"関節候補の計算に失敗しました:\n{e}",
                                     parent=win)
                q_info_var.set("計算失敗")
                return
            q_branches.clear()
            q_branches.extend(branches)
            q_state["key"] = _first_key()
            vals = [", ".join(f"{np.degrees(v):.1f}" for v in b)
                    for b in branches]
            q_combo.config(values=vals)
            if vals:
                q_combo.current(0)   # 既定 = レディ姿勢に最近傍
                q_info_var.set(
                    f"{len(vals)} 候補（先頭曲線 {names[sel[0]]} のホバー姿勢・"
                    "上ほどレディ姿勢に近い）")
                self._set_status(f"✔  スタート関節候補 {len(vals)} 件を計算しました")
            else:
                q_info_var.set("到達可能な関節候補が見つかりませんでした")
                self._set_status("⚠  スタート関節候補が見つかりませんでした")

        ttk.Button(q_row, text="候補を計算",
                   command=_compute_branches).pack(side=tk.LEFT, padx=(6, 0))

        _gen_cancel_event: list = []   # [threading.Event] — 実行中だけ存在

        def _generate():
            if not sel:
                messagebox.showwarning("曲線未選択",
                    "曲線が選択されていません。\n"
                    "3Dビューで曲線をクリックして選択してください。",
                    parent=win)
                return
            _validate_offset()   # 不正なら直前の有効な式を使用
            order = [(gi, bool(rev.get(gi, False))) for gi in sel]
            n_rev = sum(1 for _, r in order if r)
            nl = sum(1 for gi in sel if sides[gi] == "L")
            nr = len(sel) - nl
            q_start = None
            qi = q_combo.current()
            if q_branches and 0 <= qi < len(q_branches):
                q_start = q_branches[qi]

            cancel_event = threading.Event()
            _gen_cancel_event.clear()
            _gen_cancel_event.append(cancel_event)

            gen_btn.config(state="disabled")
            cancel_btn = ttk.Button(btn_row2, text="⏹ キャンセル",
                                    command=lambda: cancel_event.set())
            cancel_btn.pack(side=tk.LEFT, padx=4)
            self._set_status(
                f"選択 {len(order)} 曲線（HaL {nl} / HaR {nr}・逆方向 {n_rev}）"
                "から研磨ルート生成中 — IK計算しています"
                "（数十秒かかる場合があります）...")

            result_box: list = []   # [("ok", result) | ("cancelled",) | ("error", e)]

            def _worker():
                try:
                    r = generate_kenma_programs(
                        pts, nrm, T_blade, T_contact, self.kin,
                        selected_groups=order,
                        tool_offset=off_state["mat"],
                        q_start=q_start,
                        cancel_event=cancel_event)
                    result_box.append(("ok", r))
                except GenerationCancelled:
                    result_box.append(("cancelled",))
                except Exception as e:
                    result_box.append(("error", e))

            t = threading.Thread(target=_worker, daemon=True)
            t.start()

            def _poll():
                if t.is_alive():
                    self.root.after(150, _poll)
                    return
                # スレッド完了 — UI を復元
                _gen_cancel_event.clear()
                try:
                    cancel_btn.destroy()
                except tk.TclError:
                    pass
                gen_btn.config(state="normal")

                if not result_box:
                    return
                kind = result_box[0][0]

                if kind == "cancelled":
                    self._set_status("曲線選択ルート: キャンセルしました")
                    return

                if kind == "error":
                    e = result_box[0][1]
                    messagebox.showerror("生成エラー",
                                         f"生成に失敗しました:\n{e}", parent=win)
                    self._set_status("⚠  曲線選択ルートの生成に失敗しました")
                    return

                result = result_box[0][1]

                # IK 到達不能が全点なら早期に警告して終了
                total_wps = (len(result.route_left.waypoints)
                             + len(result.route_right.waypoints))
                if total_wps > 0 and result.n_unreachable >= total_wps:
                    messagebox.showwarning(
                        "ルート生成失敗",
                        "選択した曲線の全接触点が IK 到達不能でした。\n\n"
                        "考えられる原因:\n"
                        "  ・「パスからツールへのオフセット」の角度が大きすぎる\n"
                        "  ・UF9 STONE の位置が可動範囲外にある\n"
                        "  ・選択した曲線のグループが可動範囲外\n\n"
                        "オフセット式や砥石位置を修正してから再試行してください。",
                        parent=win)
                    self._set_status("⚠  IK 到達不能: ルートを生成できませんでした")
                    return

                _save_dlg_state()   # オフセット式・選択順を保持（次回/再生成用）

                ans = messagebox.askyesnocancel(
                    "曲線選択ルート生成",
                    f"選択 {len(order)} 曲線から生成しました。\n"
                    f"  HaL（左側面）: {nl} 曲線\n"
                    f"  HaR（右側面）: {nr} 曲線\n"
                    f"  IK到達不能: {result.n_unreachable} 点\n\n"
                    "「はい」: kenma形式LS（3ファイル1組）を出力して経路点リストへ展開\n"
                    "「いいえ」: 経路点リストへ展開のみ（シミュ再生用）\n"
                    "「キャンセル」: 何もしない\n\n"
                    "※ この画面は開いたままです。オフセット等を変更して\n"
                    "　「ルート生成」を押せば何度でもやり直せます（Ctrl+Z でも戻せます）。",
                    icon="question", parent=win)
                if ans is None:
                    self._set_status("曲線選択ルート: 生成結果を破棄しました"
                                     "（条件を変えて再生成できます）")
                    return
                if ans:
                    out_dir = filedialog.askdirectory(
                        title="kenma形式LS（選択曲線）の出力先フォルダを選択",
                        parent=win)
                    if out_dir:
                        try:
                            paths = export_kenma_ls(result, out_dir, self.kin,
                                                    mn_comments=[csv_name])
                            names_str = " / ".join(
                                os.path.basename(p) for p in paths)
                            self._set_status(
                                f"✔  選択曲線LS出力完了: {names_str} → {out_dir}")
                        except Exception as e:
                            messagebox.showerror("LS出力エラー",
                                                 f"出力に失敗しました:\n{e}",
                                                 parent=win)
                self._apply_kenma_sequence(result)
                self._set_status(
                    f"✔  選択 {len(order)} 曲線（HaL {nl} / HaR {nr}）の研磨ルートを"
                    f"経路点リストへ展開: {len(self.route)} 点 — "
                    "オフセットを変えて再生成も可能（Ctrl+Z で戻せます）")

            self.root.after(150, _poll)

        btn_row1 = ttk.Frame(win)
        btn_row1.pack(pady=(6, 2))
        ttk.Button(btn_row1, text="全選択（標準順）",
                   command=_select_all).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row1, text="リセット",
                   command=_reset).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row1, text="↑",  width=3,
                   command=lambda: _move(-1)).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row1, text="↓",  width=3,
                   command=lambda: _move(+1)).pack(side=tk.LEFT, padx=3)
        rev_btn = ttk.Button(btn_row1, text="⇄ 逆方向",
                             command=_toggle_reverse)
        rev_btn.pack(side=tk.LEFT, padx=3)
        _tip(rev_btn,
             "ハイライト中の曲線の掃引方向を反転します（RoboDK の「逆方向」）。\n"
             "→ = CSV順（刃元→刃先） / ← = 逆方向\n"
             "姿勢は順方向と同一のまま接触点の通過順だけが逆になります。\n"
             "リスト行の右クリックでも切り替えできます。")

        btn_row2 = ttk.Frame(win)
        btn_row2.pack(pady=(2, 10))
        gen_btn = ttk.Button(btn_row2, text="📐  ルート生成", style="Primary.TButton",
                             command=_generate)
        gen_btn.pack(side=tk.LEFT, padx=6)
        _tip(gen_btn,
             "選択した曲線から研磨ルートを生成します。\n"
             "生成後もこの画面は開いたまま — オフセット式や選択を変えて\n"
             "何度でも生成し直せます（Ctrl+Z で生成前に戻せます）。")
        ttk.Button(btn_row2, text="閉じる",
                   command=_close).pack(side=tk.LEFT, padx=6)

        win.protocol("WM_DELETE_WINDOW", _close)

        # ビューポートへ曲線を登録（クリックで _on_pick が呼ばれる）
        self.viewport.set_pick_curves(local_curves, _on_pick, blade_local=True)
        _refresh()
        self._set_status(
            f"曲線選択モード: {len(groups)} 曲線（左{n_left}/右{n_right}）を表示中 — "
            "3Dビューで曲線をクリックして研磨順に選択してください")

    def _clear_route(self):
        if messagebox.askyesno("確認", "経路点をすべて削除しますか？\n（Ctrl+Z で元に戻せます）"):
            self._push_undo("ルートをクリア")
            self.route.clear()
            self.route_editor.set_route(self.route)
            self.viewport.set_route(self.route)   # set_route が再描画する
            self._invalidate_sim_solutions()
            self._set_status("✔  経路をクリアしました")

    # ──────────────────────────────────────────────────────────────────
    # Info dialogs
    # ──────────────────────────────────────────────────────────────────

    def _show_dh_params(self):
        self._show_text_preview("DH パラメータ — Modified DH (Craig notation)", repr(self.kin.dh))

    def _show_robot_specs(self):
        dh = self.kin.dh
        specs = (
            f"FANUC LR Mate 200iD/14L  ロボット仕様\n"
            f"{'='*54}\n"
            f"  ペイロード       : {dh.PAYLOAD_KG} kg\n"
            f"  最大リーチ       : {dh.REACH_MM} mm  (手首中心まで)\n"
            f"  フランジリーチ   : {dh.REACH_MM + 80} mm  (フランジ端面まで)\n"
            f"  繰り返し精度     : ±{dh.REPEATABILITY_MM} mm\n"
            f"  ロボット質量     : {dh.WEIGHT_KG} kg\n"
            f"  コントローラ     : {dh.CONTROLLER}\n"
            f"  防塵防水         : {dh.IP_RATING}\n"
            f"\n{'='*54}\n"
            f"  {'軸':5} {'可動範囲最小':>12} {'可動範囲最大':>12} {'最大速度':>12}\n"
            f"  {'-'*46}\n"
        )
        for j in dh.joints:
            specs += f"  {j.name:5} {j.joint_min:>10.0f}°    {j.joint_max:>10.0f}°    {j.joint_max_speed:>8.0f}°/s\n"
        specs += (
            f"\n{'='*54}\n"
            f"  DHパラメータ (Modified DH / Z-up 座標系)\n"
            f"  {'軸':5} {'a (mm)':>8} {'alpha (°)':>10} {'d (mm)':>8}\n"
            f"  {'-'*36}\n"
        )
        for j in dh.joints:
            specs += f"  {j.name:5} {j.a:>8.0f}   {j.alpha:>10.0f}   {j.d:>8.0f}\n"
        self._show_text_preview("FANUC LR Mate 200iD/14L ロボット仕様", specs)

    def _show_about(self):
        messagebox.showinfo("About",
            f"FANUC LR Mate 200iD/14L\n"
            f"刃付けロボットシミュレータ  v{APP_VERSION}\n\n"
            f"Knife Sharpening Robot Simulator\n\n"
            f"Python  ·  matplotlib  ·  tkinter\n"
            f"運動学: Modified DH 法 (6-DOF, Z-up)\n"
            f"IK: 解析解 + scipy 数値フォールバック")

    def _show_text_preview(self, title: str, content: str):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("720x540")
        win.configure(bg=BG_DARK)
        txt = scrolledtext.ScrolledText(
            win, font=("Consolas", 9),
            bg="#0D1117", fg=FG_PRIMARY,
            insertbackground=FG_PRIMARY, wrap=tk.NONE,
            borderwidth=0, highlightthickness=0)
        txt.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        txt.insert(tk.END, content)
        txt.config(state="disabled")
        ttk.Button(win, text="閉じる", command=win.destroy).pack(pady=6)

    # ──────────────────────────────────────────────────────────────────
    # Status bar
    # ──────────────────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        self._status_var.set(msg)

    # ──────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────

    def _on_close(self):
        self._sim_running = False
        self.viewport.destroy()
        self.root.destroy()

    def run(self):
        self.root.mainloop()
