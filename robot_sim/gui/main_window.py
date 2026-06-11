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

import os
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from typing import Optional

import numpy as np

from ..robot.kinematics import Kinematics
from ..robot.dh_params import DHParams
from ..robot.tool_frame import ToolFrame
from ..robot.user_frame import UserFrame
from ..path.route import Route, Waypoint, MotionType
from ..path.csv_io import RouteCSVIO
from ..path.tp_exporter import TPExporter
from ..path.route_generator import SharpeningParams, generate_sharpening_route
from .viewport import Viewport3D
from .route_editor import RouteEditor
from .changelog import show_changelog, APP_VERSION, CHANGELOG

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


# ── ツールチップ ────────────────────────────────────────────────────────

class _Tooltip:
    """マウスホバーで説明を表示するツールチップ。"""
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
        tk.Label(
            tw, text=self._text, justify="left",
            background="#1C2333", foreground="#E6EDF3",
            relief="solid", borderwidth=1,
            font=("Yu Gothic UI", 8),
            wraplength=340, padx=8, pady=5,
        ).pack()

    def _hide(self, event=None):
        if self._win:
            self._win.destroy()
            self._win = None


def _tip(widget: tk.Widget, text: str) -> _Tooltip:
    """ウィジェットにツールチップを設定して返す。"""
    return _Tooltip(widget, text)


class MainWindow:
    """Top-level application window."""

    APP_TITLE = "FANUC LR Mate 200iD/14L  ｜  刃付けロボットシミュレータ"
    MIN_WIDTH  = 1560
    MIN_HEIGHT = 860

    TOOL_FRAMES  = [ToolFrame.flange(), ToolFrame.default_knife()]
    USER_FRAMES  = [UserFrame.world(), UserFrame.default_stone()]

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
        try:
            self._load_tormek_stl()
        except Exception:
            pass

        # 既定の文字サイズ（中）を全体に適用
        self._set_font_scale(self._font_scale)

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

    def _set_font_scale(self, scale: float):
        """UI全体の文字サイズを倍率 scale で再設定する（小=1.1/中=1.3/大=1.6）。"""
        import tkinter.font as tkfont
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
        f.add_command(label="  ✕   終了", command=self._on_close)

        # ルート
        r = menu("  ルート (Route)  ")
        r.add_command(label="  🔪  研磨経路CSVを読み込む (kenma形式)", command=self._load_kenma_route)
        r.add_command(label="  📋  基本サンプルルートを読み込む",       command=self._load_sample_route)
        r.add_command(label="  ⚙   刃付けルートを自動生成...",          command=self._auto_generate_route)
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

    # ──────────────────────────────────────────────────────────────────
    # Main panels (3D viewport + route editor)
    # ──────────────────────────────────────────────────────────────────

    def _build_main_panels(self):
        # 右パネルを root に直接貼り付け（縦いっぱい）
        right_outer = ttk.Frame(self.root, width=430)
        right_outer.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 6), pady=(4, 0))
        right_outer.pack_propagate(False)

        # ── ツリーパネル（RoboDK風・左サイド） ──────────────────────────
        tree_outer = ttk.Frame(self.root, width=220)
        tree_outer.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0), pady=(4, 0))
        tree_outer.pack_propagate(False)
        self._build_tree_panel(tree_outer)

        # ── 上部：固定フレーム（スクロール廃止 — フォント変更時のずれを防止）──
        right = ttk.Frame(right_outer)
        right.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ── 下部：折りたたみ式更新履歴 ──
        self._build_changelog_panel_collapsible(right_outer)

        # 左コンテナ：ビューポート＋ジョグパネルをまとめて右パネルと同幅に
        left_container = ttk.Frame(self.root)
        left_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0), pady=(4, 0))

        # ジョグパネルを左コンテナ下部に（先にpackしてスペース確保）
        self._build_joint_jog_panel(left_container)

        # ワークフローバー（ジョグの上・ビューポートの下）
        self._build_workflow_bar(left_container)

        # 3D ビューポートは残りの全スペースを使う
        left = ttk.LabelFrame(left_container, text="  3D ビューポート — 左ドラッグ: 回転  /  右・中ドラッグ: パン  /  ホイール: カーソル位置へズーム  /  STL・CSV をドロップで読込")
        left.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.viewport = Viewport3D(left, self.kin)

        self._build_markers_panel(right)
        self._build_ref_frames_panel(right)

        route_lf = ttk.LabelFrame(right,
            text="  経路点リスト (Waypoint List) — 追加・編集・削除・並べ替えが可能")
        route_lf.pack(fill=tk.X, padx=4, pady=(4, 2))
        self.route_editor = RouteEditor(
            route_lf, self.route,
            on_change=self._on_route_changed,
            on_select=self._on_waypoint_selected,
            listbox_height=7,
        )
        self.route_editor.pack(fill=tk.X, padx=4, pady=4)

        self._build_overlay_panel(right)

        if _HAS_DND:
            self.viewport.canvas_widget.drop_target_register(DND_FILES)
            self.viewport.canvas_widget.dnd_bind("<<Drop>>", self._on_viewport_drop)

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
            activestyle="none",
        )
        self._mk_listbox.pack(fill=tk.X)
        sb.config(command=self._mk_listbox.yview)
        self._mk_listbox.bind("<<ListboxSelect>>", self._on_mk_select)

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
            ent = ttk.Entry(pos_row, textvariable=v, width=7)
            ent.pack(side=tk.LEFT, padx=(0, 3))
            ent.bind("<MouseWheel>",
                     lambda e, i=axis_i: self._mk_scroll(e, i))
            ent.bind("<Button-4>",
                     lambda e, i=axis_i: self._mk_scroll(e, i))
            ent.bind("<Button-5>",
                     lambda e, i=axis_i: self._mk_scroll(e, i))
            ent.bind("<Return>", lambda e: self._mk_apply_pos())
        ttk.Button(pos_row, text="適用", style="Primary.TButton",
                   command=self._mk_apply_pos).pack(side=tk.LEFT, padx=2)

    def _mk_add_tcp(self):
        self._mk_tcp_count += 1
        pos = self._mk_current_tcp_pos()
        name = f"TCP-{self._mk_tcp_count}"
        self._mk_list.append({"type": "tcp", "name": name, "pos": list(pos)})
        self._mk_refresh_listbox(select_last=True)
        self._mk_sync_viewport()
        self._set_status(f"✔  TCPマーカー追加: {name}  ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})")

    def _mk_add_target(self):
        self._mk_tgt_count += 1
        pos = self._mk_current_tcp_pos()
        name = f"Target-{self._mk_tgt_count}"
        self._mk_list.append({"type": "target", "name": name, "pos": list(pos)})
        self._mk_refresh_listbox(select_last=True)
        self._mk_sync_viewport()
        self._set_status(f"✔  ターゲット追加: {name}  ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})")

    def _mk_delete(self):
        sel = self._mk_listbox.curselection()
        if not sel:
            self._set_status("⚠  削除するマーカーをリストから選択してください")
            return
        idx = sel[0]
        name = self._mk_list[idx]["name"]
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

    def _mk_scroll(self, event, axis_idx: int):
        """マウスホイールで X/Y/Z 値を増減し、即座にビューポートへ反映する。"""
        # ステップ幅: Ctrl=10mm, Shift=0.1mm, 通常=1mm
        ctrl  = bool(event.state & 0x4)
        shift = bool(event.state & 0x1)
        step  = 10.0 if ctrl else (0.1 if shift else 1.0)

        # 上スクロール判定（Windows: delta>0, Linux: Button-4）
        if hasattr(event, "delta") and event.delta != 0:
            direction = 1 if event.delta > 0 else -1
        else:
            direction = 1 if event.num == 4 else -1

        try:
            current = float(self._mk_pos_vars[axis_idx].get())
        except ValueError:
            current = 0.0
        self._mk_pos_vars[axis_idx].set(f"{current + direction * step:.2f}")

        # 選択中のマーカーがあればリアルタイム更新
        if self._mk_listbox.curselection():
            self._mk_apply_pos()
        return "break"

    def _on_mk_select(self, event=None):
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
            self.viewport.remove_ref_frame(name)
            self._rf_refresh_listbox()
            self._set_status(f"✔  参照フレーム削除: {name}")

    def _rf_clear_all(self):
        self.viewport.clear_ref_frames()
        self._rf_refresh_listbox()
        self._set_status("✔  参照フレームをすべてクリア")

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
            ttk.Entry(row, textvariable=v, width=18).pack(side=tk.LEFT, padx=4)

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

    def _build_joint_jog_panel(self, parent=None):
        """関節角度スライダーとジョグ操作を1パネルに統合。"""
        if parent is None:
            parent = self.root
        outer = ttk.Frame(parent)
        outer.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=(4, 0))

        # ---- 関節スライダー + ジョグボタン（統合パネル） ----
        slider_lf = ttk.LabelFrame(outer, text="  関節角度 / ジョグ操作")
        slider_lf.pack(side=tk.LEFT, fill=tk.Y)

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
        self._fk_display_var  = tk.StringVar()

        joint_labels = ["J1\n(旋回)", "J2\n(肩)", "J3\n(肘)", "J4\n(前腕)", "J5\n(手首↑↓)", "J6\n(手首回転)"]
        cart_labels  = ["X\n(前後)", "Y\n(左右)", "Z\n(上下)", "Rx\n(ロール)", "Ry\n(ピッチ)", "Rz\n(ヨー)"]
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
                           variable=var, orient=tk.HORIZONTAL, length=240,
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
            cart = ["X", "Y", "Z", "Rx", "Ry", "Rz"]
            for idx, (lbl, jname, cname) in enumerate(self._jog_axis_labels):
                lbl.config(text=jname if mode == "Joint" else cname)
        self._jog_mode.trace_add("write", _update_jog_labels)

        # ---- 中間列：ファイル I/O + シミュレーション + IK ----
        mid_col = ttk.Frame(outer)
        mid_col.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0))

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
        self._sim_progress_var = tk.StringVar(value="待機中")
        tk.Label(sim_inner, textvariable=self._sim_progress_var,
                 bg=BG_PANEL, fg=FG_SUB, font=("", 7)).pack()

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
        right_col.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0))

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
             "・WORLD: ロボット基準座標（デフォルト）\n"
             "・STONE: 砥石座標系（X=400mm前方, Z=200mm上方）\n"
             "3Dビューの紫色の座標軸で位置を確認できます。\n"
             "ロボット メニューから位置を編集できます。")
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
        self._joint_angles[joint_idx] = np.deg2rad(value_deg)
        self._update_viewport_from_angles(self._joint_angles)
        self._update_fk_display()

    def _on_utool_change(self, event=None):
        idx = self._utool_combo.current()
        self._active_tool = self.TOOL_FRAMES[idx]
        self.viewport.set_tool_frame(self._active_tool)
        self._set_status(f"✔  UTool 変更 → {self._active_tool.name}  "
                         f"(TCP オフセット: Z={self._active_tool.z:.0f}mm)")

    def _on_uframe_change(self, event=None):
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
            ent = ttk.Entry(inner_stl, textvariable=v, width=6)
            ent.grid(row=r, column=c*2+1, padx=1, pady=1)
            ent.bind("<MouseWheel>",
                     lambda e, idx=i: self._stl_scroll(e, idx))
            ent.bind("<Button-4>",
                     lambda e, idx=i: self._stl_scroll(e, idx))
            ent.bind("<Button-5>",
                     lambda e, idx=i: self._stl_scroll(e, idx))
            ent.bind("<Return>", lambda e: self._apply_stl_pose())
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
            ent = ttk.Entry(inner_csv, textvariable=v, width=6)
            ent.grid(row=r, column=c*2+1, padx=1, pady=1)
            ent.bind("<MouseWheel>",
                     lambda e, idx=i: self._csv_scroll(e, idx))
            ent.bind("<Button-4>",
                     lambda e, idx=i: self._csv_scroll(e, idx))
            ent.bind("<Button-5>",
                     lambda e, idx=i: self._csv_scroll(e, idx))
            ent.bind("<Return>", lambda e: self._apply_csv_pose())
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
             "デフォルト: Z=150mm (柄の先端), Rx=90° (刃渡りを刃方向に整列)\n"
             "「再読込」: 同じCSVファイルを再読込（CSV更新時に使用）")
        inner_blade = ttk.Frame(sf_blade)
        inner_blade.pack(fill=tk.X, padx=4, pady=2)
        self._blade_pose_vars: list = []
        blade_defaults = ["0.0", "0.0", "150.0", "90.0", "0.0", "0.0"]
        for i, axis in enumerate(["X", "Y", "Z", "Rx", "Ry", "Rz"]):
            r, c = divmod(i, 3)
            tk.Label(inner_blade, text=axis, bg=BG_PANEL, fg=FG_SUB,
                     font=("", 8), width=3).grid(row=r, column=c*2, padx=1)
            v = tk.StringVar(value=blade_defaults[i])
            self._blade_pose_vars.append(v)
            ent = ttk.Entry(inner_blade, textvariable=v, width=6)
            ent.grid(row=r, column=c*2+1, padx=1, pady=1)
            ent.bind("<MouseWheel>",
                     lambda e, idx=i: self._blade_scroll(e, idx))
            ent.bind("<Button-4>",
                     lambda e, idx=i: self._blade_scroll(e, idx))
            ent.bind("<Button-5>",
                     lambda e, idx=i: self._blade_scroll(e, idx))
            ent.bind("<Return>", lambda e: self._apply_blade_pose())
        btn_blade = ttk.Frame(sf_blade)
        btn_blade.pack(padx=4, pady=(0, 3))
        ttk.Button(btn_blade, text="適用", style="Primary.TButton",
                   command=self._apply_blade_pose).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_blade, text="🔄 再読込",
                   command=self._reload_blade_csv).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_blade, text="クリア",
                   command=self._clear_blade).pack(side=tk.LEFT, padx=2)

    def _overlay_step(self, event) -> float:
        ctrl  = bool(event.state & 0x4)
        shift = bool(event.state & 0x1)
        return 10.0 if ctrl else (0.1 if shift else 1.0)

    def _overlay_dir(self, event) -> int:
        if hasattr(event, "delta") and event.delta != 0:
            return 1 if event.delta > 0 else -1
        return 1 if event.num == 4 else -1

    def _stl_scroll(self, event, idx: int):
        try:
            current = float(self._stl_pose_vars[idx].get())
        except ValueError:
            current = 0.0
        self._stl_pose_vars[idx].set(
            f"{current + self._overlay_dir(event) * self._overlay_step(event):.2f}")
        self._apply_stl_pose()
        return "break"

    def _csv_scroll(self, event, idx: int):
        try:
            current = float(self._csv_pose_vars[idx].get())
        except ValueError:
            current = 0.0
        self._csv_pose_vars[idx].set(
            f"{current + self._overlay_dir(event) * self._overlay_step(event):.2f}")
        self._apply_csv_pose()
        return "break"

    def _apply_stl_pose(self):
        try:
            vals = [float(v.get()) for v in self._stl_pose_vars]
        except ValueError:
            self._set_status("⚠  数値を入力してください")
            return
        self.viewport.set_stl_pose(*vals)
        self._set_status(
            f"✔  STL 位置更新: X={vals[0]:.1f} Y={vals[1]:.1f} Z={vals[2]:.1f}")

    def _apply_csv_pose(self):
        try:
            vals = [float(v.get()) for v in self._csv_pose_vars]
        except ValueError:
            self._set_status("⚠  数値を入力してください")
            return
        self.viewport.set_csv_pose(*vals)
        self._set_status(
            f"✔  CSV 位置更新: X={vals[0]:.1f} Y={vals[1]:.1f} Z={vals[2]:.1f}")

    def _clear_stl(self):
        self.viewport.clear_stl()
        for v in self._stl_pose_vars:
            v.set("0.0")
        self._set_status("✔  STL オーバーレイをクリアしました")

    def _clear_csv(self):
        self.viewport.clear_csv()
        for v in self._csv_pose_vars:
            v.set("0.0")
        self._set_status("✔  CSV オーバーレイをクリアしました")

    # ── 刃先CSV ──────────────────────────────────────────────────────

    def _blade_scroll(self, event, idx: int):
        try:
            current = float(self._blade_pose_vars[idx].get())
        except ValueError:
            current = 0.0
        self._blade_pose_vars[idx].set(
            f"{current + self._overlay_dir(event) * self._overlay_step(event):.2f}")
        self._apply_blade_pose()
        return "break"

    def _apply_blade_pose(self):
        try:
            vals = [float(v.get()) for v in self._blade_pose_vars]
        except ValueError:
            self._set_status("⚠  数値を入力してください")
            return
        self.viewport.set_blade_pose(*vals)
        self._set_status(
            f"✔  刃先CSV 取付位置更新: Z={vals[2]:.1f} Rx={vals[3]:.1f}°")

    def _clear_blade(self):
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
                self.route.waypoints = loaded.waypoints
                self.route.name      = loaded.name
                self.route.comment   = loaded.comment
                self.route_editor.set_route(self.route)
                self._on_route_changed()
                self._set_status(
                    f"✔  経路CSV読込: {len(self.route)} 点 ← {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("読込エラー", f"CSV 読込に失敗しました:\n{e}")

    def _apply_overlay(self, kind: str):
        if kind == "stl":
            self._apply_stl_pose()
        else:
            self._apply_csv_pose()

    # ──────────────────────────────────────────────────────────────────
    # RoboDK風 ツリーパネル
    # ──────────────────────────────────────────────────────────────────

    def _build_tree_panel(self, parent):
        """Station ツリーパネルを構築する。"""
        lf = ttk.LabelFrame(parent, text="  ステーション")
        lf.pack(fill=tk.BOTH, expand=True, padx=2, pady=(2, 2))

        tree_frame = ttk.Frame(lf)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        sb = tk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._tree = ttk.Treeview(tree_frame, show="tree", selectmode="browse",
                                   yscrollcommand=sb.set)
        self._tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        sb.config(command=self._tree.yview)

        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self._tree.bind("<Double-Button-1>",  self._on_tree_double_click)
        self._tree.bind("<Button-3>",          self._on_tree_right_click)

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
            # ツリー幅に合わせた短い表示
            tree.insert(frames, "end", iid=f"uf_{uf.number}",
                         text=f"UF{uf.number}: {uf.name}"
                              f" ({uf.x:.0f},{uf.y:.0f},{uf.z:.0f})")
        # UF9 STONE – 固定参照エントリ（USER_FRAMES に含まれていない場合）
        if not any(uf.number == 9 for uf in self.USER_FRAMES):
            tree.insert(frames, "end", iid="uf_9",
                         text="UF9: STONE ← 右クリックで設定")

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

        # Targets (経路点)
        n = len(self.route.waypoints)
        targets = tree.insert(robot, "end", iid="targets",
                               text=f"🎯 Targets  ({n}点)",
                               open=(n <= 30))
        for i, wp in enumerate(self.route.waypoints):
            lbl = wp.label or f"P[{i+1}]"
            tree.insert(targets, "end", iid=f"wp_{i}", text=f"● {lbl}")

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

    def _on_tree_double_click(self, event=None):
        sel = self._tree.selection()
        if not sel:
            return
        iid = sel[0]
        if iid.startswith("wp_"):
            idx = int(iid[3:])
            if 0 <= idx < len(self.route.waypoints):
                self._ik_wp_var.set(idx + 1)
                self._compute_ik_for_wp()

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
                    label="🏗  UF9 STONE を自動設定 (x550,y-10,STL Z)",
                    command=self._setup_stone_uframe)
                menu.add_separator()
            menu.add_command(label="📐  UFrame 編集...",
                             command=self._edit_user_frame)
        elif item.startswith("ut_"):
            menu.add_command(label="🔑  UTool 編集...",
                             command=self._edit_tool_frame)
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
        path = os.path.realpath(path)
        if not os.path.isfile(path):
            messagebox.showerror("LS 読込エラー", "ファイルが見つかりません")
            return
        if not path.lower().endswith(".ls"):
            messagebox.showerror("LS 読込エラー", "LS ファイル (.ls) を選択してください")
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
                for r in routes:
                    self._tree_programs.append((r.name, r))
                self._tree_refresh()
                self._set_status(
                    f"✔  LS 読込 (ツリー追加): {len(routes)} プログラム  計 {total} 点")

    def _apply_prog_route(self, route):
        """読み込んだ Route を現在の経路として適用する。"""
        self.route.waypoints = list(route.waypoints)
        self.route.name      = route.name
        self.route.comment   = route.comment
        if route.uframe:
            self.route.uframe = route.uframe
        if route.utool:
            self.route.utool  = route.utool
        self.route_editor.set_route(self.route)
        self.viewport.set_route(self.route)
        self.viewport.refresh()
        self._tree_refresh()
        self._set_status(
            f"✔  経路適用: {route.name}  {len(self.route)} 点")

    def _remove_prog(self, idx: int):
        if 0 <= idx < len(self._tree_programs):
            del self._tree_programs[idx]
            self._tree_refresh()

    # ──────────────────────────────────────────────────────────────────
    # UF9 STONE 自動設定
    # ──────────────────────────────────────────────────────────────────

    def _setup_stone_uframe(self):
        """STL bbox の Z 最大値を grinder_top_z として UF9 STONE を自動設定する。"""
        bb = self.viewport.stl_bbox()
        if bb and self.viewport._stl_verts is not None:
            R = self.viewport._stl_T[:3, :3]
            t = self.viewport._stl_T[:3, 3]
            all_v = self.viewport._stl_verts.reshape(-1, 3)
            tv = ((R @ all_v.T).T + t)
            grinder_top_z = float(tv[:, 2].max())
        else:
            grinder_top_z = 332.0

        from ..robot.user_frame import UserFrame as _UF
        uf9 = _UF(number=9, name="STONE",
                  x=550.0, y=-10.0, z=grinder_top_z,
                  rx=0.0,  ry=0.0,  rz=90.0,
                  comment="Grinder top surface")

        nums = [uf.number for uf in self.USER_FRAMES]
        if 9 in nums:
            self.USER_FRAMES[nums.index(9)] = uf9
        else:
            self.USER_FRAMES.append(uf9)

        # コンボボックスを更新
        uf_names = [f"UF{u.number}: {u.name}" for u in self.USER_FRAMES]
        self._uframe_combo.config(values=uf_names)
        idx9 = next(i for i, uf in enumerate(self.USER_FRAMES) if uf.number == 9)
        self._uframe_combo.current(idx9)
        self._active_uframe = self.USER_FRAMES[idx9]
        self.viewport.set_user_frame(self._active_uframe)

        self._tree_refresh()
        self._set_status(
            f"✔  UF9 STONE 設定: x=550, y=-10, z={grinder_top_z:.0f}mm, rz=90°")

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
            tk.Label(frame, text=text, bg=BG_PANEL, fg=ACCENT2,
                     font=("Yu Gothic UI", 8, "bold")).pack(anchor="w", pady=(8, 2))

        def row(label, default, hint=""):
            f = ttk.Frame(frame)
            f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, bg=BG_PANEL, fg=FG_PRIMARY,
                     font=("", 8), width=22, anchor="w").pack(side=tk.LEFT)
            var = tk.StringVar(value=str(default))
            ttk.Entry(f, textvariable=var, width=8).pack(side=tk.LEFT)
            if hint:
                tk.Label(f, text=hint, bg=BG_PANEL, fg=FG_SUB,
                         font=("", 7)).pack(side=tk.LEFT, padx=4)
            return var

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
            ok = self.viewport.load_stl(path)
            if ok:
                self._set_status(f"✔  STL 読込: {os.path.basename(path)}")
            else:
                self._set_status("⚠  STL 読込失敗: numpy-stl が必要です (pip install numpy-stl)")
        elif ext == ".csv":
            if self._is_blade_csv(path):
                self._load_blade_csv(path)
            else:
                ok = self.viewport.load_csv_points(path)
                if ok:
                    self._set_status(f"✔  CSV 読込: {os.path.basename(path)}")
                else:
                    self._set_status("⚠  CSV に有効な X,Y,Z 列がありません")
        else:
            self._set_status(f"⚠  対応形式: .stl または .csv のみ ({ext})")

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
        text = (
            f"  位置 X: {x:8.1f} mm    姿勢 Rx: {rx:7.1f} °\n"
            f"  位置 Y: {y:8.1f} mm    姿勢 Ry: {ry:7.1f} °\n"
            f"  位置 Z: {z:8.1f} mm    姿勢 Rz: {rz:7.1f} °"
        )
        self._fk_display_var.set(
            f"Pos: ({x:7.1f}, {y:7.1f}, {z:7.1f}) mm   "
            f"RPY: ({rx:6.1f}, {ry:6.1f}, {rz:6.1f}) °"
        )
        if hasattr(self, "_fk_detail_var"):
            self._fk_detail_var.set(text)

    # ──────────────────────────────────────────────────────────────────
    # Route events
    # ──────────────────────────────────────────────────────────────────

    def _on_route_changed(self):
        self.viewport.set_route(self.route)
        self.viewport.refresh()
        n   = len(self.route)
        self._set_status(f"✔  経路更新 — {n} 点")
        if hasattr(self, "_tree"):
            self._tree_refresh()

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
        path = os.path.realpath(path)
        if not os.path.isfile(path):
            messagebox.showerror("読込エラー", "ファイルが見つかりません")
            return
        if not path.lower().endswith(".csv"):
            messagebox.showerror("読込エラー", "CSV ファイル (.csv) を選択してください")
            return
        try:
            loaded = RouteCSVIO.route_from_csv(path)
            self.route.waypoints = loaded.waypoints
            self.route.name      = loaded.name
            self.route.comment   = loaded.comment
            self.route_editor.set_route(self.route)
            self.viewport.set_route(self.route)
            self.viewport.refresh()
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
                # UF9: X=550,Y=-10,Z=300(table),W=0,P=0,R=90
                uframe_pos = (550.0, -10.0, 300.0, 0.0, 0.0, 90.0)
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
    # Simulation
    # ──────────────────────────────────────────────────────────────────

    def _start_simulation(self):
        if not self.route.waypoints:
            messagebox.showwarning("経路点なし", "経路点が1つもありません。")
            return
        if self._sim_thread and self._sim_thread.is_alive():
            return
        self._sim_running = True
        self._sim_btn.config(state="disabled")
        override = self._speed_override.get() / 100.0
        total    = len(self.route.waypoints)

        def run():
            q_prev    = self._joint_angles.copy()
            waypoints = list(self.route.waypoints)
            for i, wp in enumerate(waypoints):
                if not self._sim_running:
                    break
                T = wp.to_transform()
                q_target, ok = self.kin.inverse(T, q_init=q_prev)
                if not ok:
                    self.root.after(0, lambda i=i: self._set_status(
                        f"⚠  IK 失敗: P[{i+1}] — 可動範囲外の可能性があります"))
                    q_target = q_prev

                speeds_rad = np.deg2rad(self.kin.dh.get_joint_max_speeds()) * override
                delta      = np.abs(q_target - q_prev)
                max_time   = float(np.max(delta / np.maximum(speeds_rad, 1e-6)))
                steps      = max(20, int(max_time / 0.03))

                for step in range(steps + 1):
                    if not self._sim_running:
                        break
                    alpha   = step / steps
                    q_interp = q_prev + alpha * (q_target - q_prev)

                    def _update(q=q_interp.copy(), idx=i):
                        self._joint_angles = q
                        self._update_viewport_from_angles(q)
                        self._update_fk_display()
                        for j, var in enumerate(self._slider_vars):
                            var.set(np.rad2deg(q[j]))
                        self.viewport.set_selected_waypoint(idx)
                        pct = int((idx + alpha) / total * 100)
                        self._sim_progress_var.set(f"P[{idx+1}]/{total}  {pct}%")
                        self._set_status(
                            f"▶  シミュレーション実行中 — P[{idx+1}/{total}]  {wp.label}  "
                            f"({pct}%)")

                    self.root.after(0, _update)
                    time.sleep(0.03)

                q_prev = q_target

            self.root.after(0, self._simulation_done)

        self._sim_thread = threading.Thread(target=run, daemon=True)
        self._sim_thread.start()

    def _stop_simulation(self):
        self._sim_running = False

    def _simulation_done(self):
        self._sim_running = False
        self._sim_btn.config(state="normal")
        self.viewport.set_selected_waypoint(None)
        self.viewport.set_jog_target(None)
        self._sim_progress_var.set("完了")
        self._set_status("✔  シミュレーション完了")

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
        self._set_angles(self.kin.dh.home_position())
        self._set_status("✔  ホームポジション (全軸 0°) に移動しました")

    def _go_ready(self):
        self._set_angles(self.kin.dh.ready_position())
        self._set_status("✔  レディポジション (J2=-45° J3=+30° J5=-60°) に移動しました")

    def _set_angles(self, q: np.ndarray):
        self._joint_angles = q.copy()
        for i, var in enumerate(self._slider_vars):
            var.set(np.rad2deg(q[i]))
        self._update_viewport_from_angles(q)
        self._update_fk_display()

    # ──────────────────────────────────────────────────────────────────
    # Tool / User frame dialogs
    # ──────────────────────────────────────────────────────────────────

    def _edit_tool_frame(self):
        tf = self._active_tool
        self._frame_editor_dialog(
            title=f"ツールフレーム編集: {tf.name}",
            desc="フランジ（J6先端）からTCP（ツール中心点）までのオフセットを設定します。\n包丁の場合、刃の中心まで Z方向に延長します。",
            obj=tf,
            fields=["x", "y", "z", "rx", "ry", "rz"],
            labels=["X オフセット (mm)", "Y オフセット (mm)", "Z オフセット (mm)",
                    "Rx 回転 (°)", "Ry 回転 (°)", "Rz 回転 (°)"],
            on_apply=lambda: self.viewport.set_tool_frame(self._active_tool)
        )

    def _edit_user_frame(self):
        uf = self._active_uframe
        self._frame_editor_dialog(
            title=f"ユーザーフレーム編集: {uf.name}",
            desc="作業座標系の原点を設定します。\n砥石の場合、砥石面の中心をユーザーフレーム原点とします。",
            obj=uf,
            fields=["x", "y", "z", "rx", "ry", "rz"],
            labels=["X 位置 (mm)", "Y 位置 (mm)", "Z 位置 (mm)",
                    "Rx 回転 (°)", "Ry 回転 (°)", "Rz 回転 (°)"],
            on_apply=lambda: self.viewport.set_user_frame(self._active_uframe)
        )

    def _frame_editor_dialog(self, title, desc, obj, fields, labels, on_apply):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("380x340")
        win.configure(bg=BG_DARK)
        win.resizable(False, False)

        tk.Label(win, text=title, bg=BG_DARK, fg=ACCENT,
                 font=("Yu Gothic UI", 10, "bold")).pack(pady=(12, 2), padx=12, anchor="w")
        tk.Label(win, text=desc, bg=BG_DARK, fg=FG_SUB,
                 font=("Yu Gothic UI", 8), justify="left",
                 wraplength=350).pack(padx=12, anchor="w")

        ttk.Separator(win).pack(fill=tk.X, padx=12, pady=8)

        vars_ = {}
        for f, lbl in zip(fields, labels):
            row = ttk.Frame(win)
            row.pack(fill=tk.X, padx=16, pady=2)
            tk.Label(row, text=lbl, bg=BG_PANEL, fg=FG_SUB,
                     font=("", 8), width=18, anchor="w").pack(side=tk.LEFT)
            v = tk.StringVar(value=str(getattr(obj, f)))
            vars_[f] = v
            ttk.Entry(row, textvariable=v, width=12).pack(side=tk.LEFT, padx=4)

        def apply():
            for f, v in vars_.items():
                try:
                    setattr(obj, f, float(v.get()))
                except ValueError:
                    pass
            on_apply()
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
        assets = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets")
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
            self.route.waypoints = loaded.waypoints
            self.route.name      = loaded.name or os.path.splitext(os.path.basename(path))[0]
            self.route.comment   = loaded.comment or "Knife sharpening route"
            if loaded.uframe:
                self.route.uframe = loaded.uframe
            if loaded.utool:
                self.route.utool = loaded.utool
            self.route_editor.set_route(self.route)
            self.viewport.set_route(self.route)
            self.viewport.refresh()
            self._set_status(
                f"✔  研磨経路読込完了: {len(self.route)} 点 "
                f"(UF{self.route.uframe}/UT{self.route.utool}) ← {os.path.basename(path)}"
            )
        except Exception as e:
            messagebox.showerror("読込エラー", f"研磨経路CSV の読込に失敗しました:\n{e}")

    def _load_sample_route(self):
        sample = Route.default_sharpening_route()
        self.route.waypoints = sample.waypoints
        self.route.name      = sample.name
        self.route.comment   = sample.comment
        self.route.uframe    = self._active_uframe.number
        self.route.utool     = self._active_tool.number
        self.route_editor.set_route(self.route)
        self.viewport.set_route(self.route)
        self.viewport.refresh()
        self._set_status(f"✔  サンプルルート読込完了 — {len(self.route)} 点")

    # 研磨機 STL の既定配置パラメータ（ユーザー確認済みの値）
    _STL_DEFAULT_POSE = (800.0, 148.0, 266.0, 0.0, 0.0, -90.0)

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
        assets = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets")
        stl_path = os.path.realpath(os.path.join(assets, "Tormek_T8.stl"))
        if not os.path.isfile(stl_path):
            self._set_status("⚠  STL ファイルが見つかりません: " + stl_path)
            return
        ok = self.viewport.load_stl(stl_path)
        if ok:
            self._apply_stl_default_pose()
            # Auto-add UF9 stone top reference frame using actual STL bbox
            bb = self.viewport.stl_bbox()
            if bb and self.viewport._stl_verts is not None:
                R = self.viewport._stl_T[:3, :3]
                t = self.viewport._stl_T[:3, 3]
                all_v = self.viewport._stl_verts.reshape(-1, 3)
                tv = ((R @ all_v.T).T + t)
                stone_top_z = float(tv[:, 2].max())
            else:
                stone_top_z = 266.0
            self.viewport.remove_ref_frame("UF9: STONE")
            self.viewport.add_ref_frame(
                "UF9: STONE", 550, -10, stone_top_z, 0, 0, 90, color="#FF88FF")
            self._set_status(
                f"✔  Tormek T8 STL 読込済（X=800, Y=148, Z=266, Rz=-90°）  UF9 STONE z={stone_top_z:.0f}mm")
        else:
            self._set_status("⚠  STL 読込失敗")

    def _load_tormek_csv(self):
        """Tormek 研削経路 CSV のみ読み込む（STLは触らない）。"""
        assets = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets")
        csv_path = os.path.join(assets, "grinding_path_sample.csv")
        if not os.path.exists(csv_path):
            self._set_status("⚠  研削経路 CSV が見つかりません: " + csv_path)
            return
        ok = self.viewport.load_csv_points(csv_path)
        if ok:
            self._set_status("✔  Tormek 研削経路 CSV 読込済")
        else:
            self._set_status("⚠  CSV 読込失敗")

    def _load_tormek_sample(self):
        """後方互換: STL + CSV を両方読み込む（起動時の自動ロード用）。"""
        self._load_tormek_stl()
        self._load_tormek_csv()

    def _auto_generate_route(self):
        win = tk.Toplevel(self.root)
        win.title("刃付けルート自動生成")
        win.geometry("460x560")
        win.configure(bg=BG_DARK)
        win.resizable(False, False)

        tk.Label(win, text="⚙  刃付けルート自動生成",
                 bg=BG_DARK, fg=ACCENT,
                 font=("Yu Gothic UI", 12, "bold")).pack(pady=(14, 2), padx=16, anchor="w")
        tk.Label(win,
            text="砥石の位置・寸法と刃付けパラメータを入力すると、\n往復研磨ルートを自動で生成します。",
            bg=BG_DARK, fg=FG_SUB,
            font=("Yu Gothic UI", 8), justify="left").pack(padx=16, anchor="w")

        ttk.Separator(win).pack(fill=tk.X, padx=16, pady=8)

        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=16)

        def section(text):
            tk.Label(frame, text=text, bg=BG_PANEL, fg=ACCENT2,
                     font=("Yu Gothic UI", 8, "bold")).pack(anchor="w", pady=(8, 2))

        def row(label, default, hint=""):
            f = ttk.Frame(frame)
            f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, bg=BG_PANEL, fg=FG_PRIMARY,
                     font=("", 8), width=26, anchor="w").pack(side=tk.LEFT)
            var = tk.StringVar(value=str(default))
            ttk.Entry(f, textvariable=var, width=8).pack(side=tk.LEFT)
            if hint:
                tk.Label(f, text=hint, bg=BG_PANEL, fg=FG_SUB,
                         font=("", 7)).pack(side=tk.LEFT, padx=4)
            return var

        section("▸ 砥石の位置（ロボット基準座標）")
        v_sx = row("砥石 X mm（前方距離）:", 400, "ロボット正面方向")
        v_sy = row("砥石 Y mm（左右）:",      0,   "正値=左")
        v_sz = row("砥石 Z mm（高さ）:",    250,   "床面からの高さ")

        section("▸ 砥石の寸法")
        v_slen = row("砥石の長さ mm（包丁スライド方向）:", 200)
        v_swid = row("砥石の幅  mm（包丁送り方向）:",      70)

        section("▸ 刃付けパラメータ")
        v_ang  = row("刃の角度 °（砥石面に対する傾き）:", 15, "一般的: 10〜20°")
        v_blen = row("研磨する刃の長さ mm:",              180)
        v_strk = row("往復ストローク回数:",                5)
        v_spd  = row("ストローク速度 mm/s:",              30, "推奨: 20〜50")

        def on_generate():
            try:
                p = SharpeningParams(
                    stone_x=float(v_sx.get()), stone_y=float(v_sy.get()),
                    stone_z=float(v_sz.get()),
                    stone_length=float(v_slen.get()), stone_width=float(v_swid.get()),
                    blade_angle_deg=float(v_ang.get()),
                    blade_length_mm=float(v_blen.get()),
                    num_strokes=int(v_strk.get()),
                    stroke_speed_mms=float(v_spd.get()),
                    utool=self._active_tool.number,
                    uframe=self._active_uframe.number,
                )
                new_route = generate_sharpening_route(p)
                self.route.waypoints = new_route.waypoints
                self.route.name      = new_route.name
                self.route.comment   = new_route.comment
                self.route.uframe    = p.uframe
                self.route.utool     = p.utool
                self.route_editor.set_route(self.route)
                self.viewport.set_route(self.route)
                self.viewport.refresh()
                self._set_status(
                    f"✔  ルート自動生成完了 — {len(self.route)} 点  "
                    f"(ストローク: {int(v_strk.get())} 往復 × "
                    f"{max(1, int(float(v_blen.get())/(float(v_swid.get())-10)))} パス)")
                win.destroy()
            except Exception as e:
                messagebox.showerror("生成エラー", str(e), parent=win)

        ttk.Separator(win).pack(fill=tk.X, padx=16, pady=8)
        btn_row = ttk.Frame(win)
        btn_row.pack(pady=4)
        ttk.Button(btn_row, text="⚙  ルートを生成する",
                   style="Primary.TButton",
                   command=on_generate).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_row, text="キャンセル",
                   command=win.destroy).pack(side=tk.LEFT, padx=6)

    def _clear_route(self):
        if messagebox.askyesno("確認", "経路点をすべて削除しますか？\nこの操作は元に戻せません。"):
            self.route.clear()
            self.route_editor.set_route(self.route)
            self.viewport.set_route(self.route)
            self.viewport.refresh()
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
