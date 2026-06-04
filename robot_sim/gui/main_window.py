"""
Main tkinter window for the FANUC LR Mate 200iD/14L knife sharpening simulator.

Layout:
  ┌──────────────────────────────────────────────────────────────────┐
  │  Menu bar                                                         │
  ├──────────────────────────┬───────────────────────────────────────┤
  │  3D Viewport             │  Route Editor (waypoint list)         │
  │                          │  [ Add ] [Edit] [Del] [↑] [↓]        │
  ├──────────────────────────┴───────────────────────────────────────┤
  │  Joint sliders J1-J6   |  Speed Override   |  UTool / UFrame     │
  ├──────────────────────────────────────────────────────────────────┤
  │  Jog panel  |  File I/O  |  Simulation  |  IK  |  FK result      │
  └──────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import os
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
from .changelog import show_changelog, APP_VERSION


class MainWindow:
    """Top-level application window."""

    APP_TITLE = "FANUC LR Mate 200iD/14L  |  刃付けロボットシミュレータ"
    MIN_WIDTH = 1280
    MIN_HEIGHT = 820

    # Available tool frames
    TOOL_FRAMES = [
        ToolFrame.flange(),
        ToolFrame.default_knife(),
    ]
    # Available user frames
    USER_FRAMES = [
        UserFrame.world(),
        UserFrame.default_stone(),
    ]

    def __init__(self):
        self.kin = Kinematics()
        self.route = Route.default_sharpening_route()
        self._joint_angles = self.kin.dh.ready_position().copy()
        self._sim_thread: Optional[threading.Thread] = None
        self._sim_running = False
        self._simulator = None

        self._active_tool = self.TOOL_FRAMES[1]   # KNIFE default
        self._active_uframe = self.USER_FRAMES[0]  # WORLD default

        self._build_root()
        self._build_menu()
        self._build_main_panels()
        self._build_joint_sliders()
        self._build_bottom_controls()
        self._build_status_bar()

        self.viewport.set_route(self.route)
        self.viewport.set_tool_frame(self._active_tool)
        self.viewport.set_user_frame(self._active_uframe)
        self._update_viewport_from_angles(self._joint_angles)
        self._update_fk_display()

    # ------------------------------------------------------------------
    # Root window
    # ------------------------------------------------------------------

    def _build_root(self):
        self.root = tk.Tk()
        self.root.title(self.APP_TITLE)
        self.root.minsize(self.MIN_WIDTH, self.MIN_HEIGHT)
        self.root.configure(bg="#1E1E1E")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background="#2A2A2A", foreground="#DDDDDD",
                        fieldbackground="#333333", bordercolor="#555555")
        style.configure("TButton", padding=4, relief="flat",
                        background="#3A3A3A", foreground="#DDDDDD")
        style.map("TButton", background=[("active", "#4A4A6A")])
        style.configure("TLabel", background="#2A2A2A", foreground="#DDDDDD")
        style.configure("TFrame", background="#2A2A2A")
        style.configure("TLabelframe", background="#2A2A2A", foreground="#AAAAAA")
        style.configure("TLabelframe.Label", background="#2A2A2A", foreground="#AAAAAA")
        style.configure("TEntry", fieldbackground="#333333", foreground="#DDDDDD")
        style.configure("TCombobox", fieldbackground="#333333", foreground="#DDDDDD")
        style.configure("Treeview", background="#2A2A2A", foreground="#DDDDDD",
                        fieldbackground="#2A2A2A")
        style.configure("Jog.TButton", padding=2, width=3,
                        background="#3A3A4A", foreground="#DDDDDD")
        style.map("Jog.TButton", background=[("active", "#5A5A8A")])

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu(self):
        menubar = tk.Menu(self.root, bg="#1E1E1E", fg="#DDDDDD",
                          activebackground="#4A4A6A", activeforeground="white")

        # File
        file_menu = tk.Menu(menubar, tearoff=0, bg="#2A2A2A", fg="#DDDDDD",
                            activebackground="#4A4A6A")
        file_menu.add_command(label="CSV を開く...", command=self._load_csv, accelerator="Ctrl+O")
        file_menu.add_command(label="CSV として保存...", command=self._save_csv, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="TP プログラムをエクスポート...", command=self._export_tp, accelerator="Ctrl+E")
        file_menu.add_separator()
        file_menu.add_command(label="終了", command=self._on_close)
        menubar.add_cascade(label="ファイル (File)", menu=file_menu)

        # Route
        route_menu = tk.Menu(menubar, tearoff=0, bg="#2A2A2A", fg="#DDDDDD",
                             activebackground="#4A4A6A")
        route_menu.add_command(label="サンプルルートを読み込む", command=self._load_sample_route)
        route_menu.add_command(label="刃付けルートを自動生成...", command=self._auto_generate_route)
        route_menu.add_command(label="ルートをクリア", command=self._clear_route)
        route_menu.add_separator()
        route_menu.add_command(label="▶ シミュレーション実行", command=self._start_simulation, accelerator="F5")
        menubar.add_cascade(label="ルート (Route)", menu=route_menu)

        # Robot
        robot_menu = tk.Menu(menubar, tearoff=0, bg="#2A2A2A", fg="#DDDDDD",
                              activebackground="#4A4A6A")
        robot_menu.add_command(label="ホームポジション", command=self._go_home)
        robot_menu.add_command(label="レディポジション", command=self._go_ready)
        robot_menu.add_separator()
        robot_menu.add_command(label="ツールフレームを編集...", command=self._edit_tool_frame)
        robot_menu.add_command(label="ユーザーフレームを編集...", command=self._edit_user_frame)
        robot_menu.add_separator()
        robot_menu.add_command(label="DH パラメータを表示", command=self._show_dh_params)
        robot_menu.add_command(label="ロボット仕様を表示", command=self._show_robot_specs)
        menubar.add_cascade(label="ロボット (Robot)", menu=robot_menu)

        # Help
        help_menu = tk.Menu(menubar, tearoff=0, bg="#2A2A2A", fg="#DDDDDD",
                            activebackground="#4A4A6A")
        help_menu.add_command(label=f"チェンジログ (v{APP_VERSION})...", command=lambda: show_changelog(self.root))
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="ヘルプ (Help)", menu=help_menu)

        self.root.config(menu=menubar)
        self.root.bind("<Control-o>", lambda e: self._load_csv())
        self.root.bind("<Control-s>", lambda e: self._save_csv())
        self.root.bind("<Control-e>", lambda e: self._export_tp())
        self.root.bind("<F5>", lambda e: self._start_simulation())

    # ------------------------------------------------------------------
    # Main panels
    # ------------------------------------------------------------------

    def _build_main_panels(self):
        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left_frame = ttk.LabelFrame(pane, text="3D ビューポート (3D Viewport)")
        pane.add(left_frame, weight=3)
        self.viewport = Viewport3D(left_frame, self.kin)

        right_frame = ttk.LabelFrame(pane, text="ルートエディタ (Route Editor)")
        pane.add(right_frame, weight=1)
        self.route_editor = RouteEditor(
            right_frame, self.route,
            on_change=self._on_route_changed,
            on_select=self._on_waypoint_selected,
        )
        self.route_editor.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ------------------------------------------------------------------
    # Joint sliders + Speed Override + UTool/UFrame
    # ------------------------------------------------------------------

    def _build_joint_sliders(self):
        outer = ttk.Frame(self.root)
        outer.pack(fill=tk.X, padx=4, pady=2)

        # Joint sliders
        slider_frame = ttk.LabelFrame(outer, text="ジョイント角度 (Joint Angles)")
        slider_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        lower, upper = self.kin.dh.get_joint_limits_deg()
        self._slider_vars = []

        for i in range(6):
            col = ttk.Frame(slider_frame)
            col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4, pady=2)
            init_deg = np.rad2deg(self._joint_angles[i])
            var = tk.DoubleVar(value=init_deg)
            self._slider_vars.append(var)
            ttk.Label(col, text=f"J{i+1}", width=4, anchor="center").pack()
            ttk.Scale(col, from_=lower[i], to=upper[i], variable=var,
                      orient=tk.VERTICAL, length=90,
                      command=lambda val, idx=i: self._on_slider_change(idx, float(val))
                      ).pack()
            ttk.Label(col, textvariable=var, width=6, anchor="center").pack()

        self._angles_display_var = tk.StringVar()
        ttk.Label(slider_frame, textvariable=self._angles_display_var,
                  font=("Courier", 8), foreground="#888888").pack(side=tk.RIGHT, padx=8)
        self._update_angles_display()

        # Right side: Speed Override + UTool + UFrame
        right_col = ttk.Frame(outer)
        right_col.pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # Speed override
        spd_frame = ttk.LabelFrame(right_col, text="速度オーバーライド (%)")
        spd_frame.pack(fill=tk.X, pady=2)
        self._speed_override = tk.IntVar(value=100)
        ttk.Scale(spd_frame, from_=1, to=100, variable=self._speed_override,
                  orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT, padx=4)
        ttk.Label(spd_frame, textvariable=self._speed_override, width=4).pack(side=tk.LEFT)

        # UTool selector
        tool_frame_ui = ttk.LabelFrame(right_col, text="UTool")
        tool_frame_ui.pack(fill=tk.X, pady=2)
        self._utool_var = tk.StringVar(value=self._active_tool.name)
        tool_names = [f"UT{t.number}: {t.name}" for t in self.TOOL_FRAMES]
        self._utool_combo = ttk.Combobox(tool_frame_ui, textvariable=self._utool_var,
                                          values=tool_names, state="readonly", width=14)
        self._utool_combo.current(1)
        self._utool_combo.pack(padx=4, pady=2)
        self._utool_combo.bind("<<ComboboxSelected>>", self._on_utool_change)

        # UFrame selector
        uf_frame_ui = ttk.LabelFrame(right_col, text="UFrame")
        uf_frame_ui.pack(fill=tk.X, pady=2)
        self._uframe_var = tk.StringVar(value=self._active_uframe.name)
        uf_names = [f"UF{u.number}: {u.name}" for u in self.USER_FRAMES]
        self._uframe_combo = ttk.Combobox(uf_frame_ui, textvariable=self._uframe_var,
                                           values=uf_names, state="readonly", width=14)
        self._uframe_combo.current(0)
        self._uframe_combo.pack(padx=4, pady=2)
        self._uframe_combo.bind("<<ComboboxSelected>>", self._on_uframe_change)

    def _on_slider_change(self, joint_idx: int, value_deg: float):
        self._joint_angles[joint_idx] = np.deg2rad(value_deg)
        self._update_viewport_from_angles(self._joint_angles)
        self._update_fk_display()
        self._update_angles_display()

    def _update_angles_display(self):
        deg = np.rad2deg(self._joint_angles)
        self._angles_display_var.set(
            "  ".join(f"J{i+1}:{d:+6.1f}°" for i, d in enumerate(deg))
        )

    def _on_utool_change(self, event=None):
        idx = self._utool_combo.current()
        self._active_tool = self.TOOL_FRAMES[idx]
        self.viewport.set_tool_frame(self._active_tool)
        self._set_status(f"UTool → {self._active_tool.name}")

    def _on_uframe_change(self, event=None):
        idx = self._uframe_combo.current()
        self._active_uframe = self.USER_FRAMES[idx]
        self.viewport.set_user_frame(self._active_uframe)
        self._set_status(f"UFrame → {self._active_uframe.name}")

    # ------------------------------------------------------------------
    # Bottom controls: Jog | File I/O | Sim | IK | FK
    # ------------------------------------------------------------------

    def _build_bottom_controls(self):
        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill=tk.X, padx=4, pady=4)

        # ---- Jog panel ----
        jog_outer = ttk.LabelFrame(ctrl, text="ジョグ (Jog)")
        jog_outer.pack(side=tk.LEFT, padx=4, fill=tk.Y)

        self._jog_mode = tk.StringVar(value="Joint")
        ttk.Radiobutton(jog_outer, text="Joint", variable=self._jog_mode,
                        value="Joint").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(jog_outer, text="Cartesian", variable=self._jog_mode,
                        value="Cartesian").pack(side=tk.LEFT, padx=2)

        step_frame = ttk.Frame(jog_outer)
        step_frame.pack(side=tk.LEFT, padx=4)
        ttk.Label(step_frame, text="Step:").pack(side=tk.LEFT)
        self._jog_step = tk.StringVar(value="5")
        ttk.Combobox(step_frame, textvariable=self._jog_step,
                     values=["0.5", "1", "5", "10", "45"],
                     width=5, state="readonly").pack(side=tk.LEFT)

        axes_frame = ttk.Frame(jog_outer)
        axes_frame.pack(side=tk.LEFT, padx=4)
        jog_axes = ["J1/X", "J2/Y", "J3/Z", "J4/Rx", "J5/Ry", "J6/Rz"]
        for col, label in enumerate(jog_axes):
            f = ttk.Frame(axes_frame)
            f.grid(row=0, column=col, padx=1)
            ttk.Label(f, text=label, font=("", 7), width=5, anchor="center").pack()
            ttk.Button(f, text="▲", style="Jog.TButton",
                       command=lambda ax=col: self._jog(ax, +1)).pack()
            ttk.Button(f, text="▼", style="Jog.TButton",
                       command=lambda ax=col: self._jog(ax, -1)).pack()

        # ---- File I/O ----
        io_frame = ttk.LabelFrame(ctrl, text="ファイル I/O")
        io_frame.pack(side=tk.LEFT, padx=4)
        ttk.Button(io_frame, text="CSV 読込", command=self._load_csv, width=9).pack(side=tk.LEFT, padx=2, pady=2)
        ttk.Button(io_frame, text="CSV 保存", command=self._save_csv, width=9).pack(side=tk.LEFT, padx=2, pady=2)
        ttk.Button(io_frame, text="TP 出力", command=self._export_tp, width=9).pack(side=tk.LEFT, padx=2, pady=2)

        # ---- Simulation ----
        sim_frame = ttk.LabelFrame(ctrl, text="シミュレーション")
        sim_frame.pack(side=tk.LEFT, padx=4)
        self._sim_btn = ttk.Button(sim_frame, text="▶ 実行", command=self._start_simulation, width=8)
        self._sim_btn.pack(side=tk.LEFT, padx=2, pady=2)
        ttk.Button(sim_frame, text="■ 停止", command=self._stop_simulation, width=6).pack(side=tk.LEFT, padx=2, pady=2)

        # ---- IK ----
        ik_frame = ttk.LabelFrame(ctrl, text="逆運動学 (IK)")
        ik_frame.pack(side=tk.LEFT, padx=4)
        self._ik_wp_var = tk.IntVar(value=1)
        ttk.Label(ik_frame, text="WP:").pack(side=tk.LEFT)
        ttk.Spinbox(ik_frame, from_=1, to=999, textvariable=self._ik_wp_var, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Button(ik_frame, text="IK 計算", command=self._compute_ik_for_wp, width=7).pack(side=tk.LEFT, padx=2, pady=2)

        # ---- FK result ----
        fk_frame = ttk.LabelFrame(ctrl, text="FK 結果 (mm / deg)")
        fk_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self._fk_display_var = tk.StringVar()
        ttk.Label(fk_frame, textvariable=self._fk_display_var,
                  font=("Courier", 8), foreground="#88CCFF").pack(anchor="w", padx=4)

    def _build_status_bar(self):
        self._status_var = tk.StringVar(value=f"Ready  /  準備完了   [v{APP_VERSION}]")
        ttk.Label(self.root, textvariable=self._status_var,
                  relief=tk.SUNKEN, anchor=tk.W, font=("", 8)
                  ).pack(fill=tk.X, side=tk.BOTTOM, padx=2, pady=1)

    # ------------------------------------------------------------------
    # Jog
    # ------------------------------------------------------------------

    def _jog(self, axis: int, direction: int):
        """Jog robot in joint or Cartesian mode."""
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
            # Cartesian jog: compute FK, shift position/orientation, IK
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
                self._set_status("Cartesian jog: IK 失敗 — 可動範囲外")

    # ------------------------------------------------------------------
    # Viewport & FK helpers
    # ------------------------------------------------------------------

    def _update_viewport_from_angles(self, q: np.ndarray):
        self.viewport.update_robot(q)

    def _update_fk_display(self):
        T = self.kin.forward(self._joint_angles)
        x, y, z, rx, ry, rz = self.kin.transform_to_pose(T)
        self._fk_display_var.set(
            f"Pos: ({x:7.1f}, {y:7.1f}, {z:7.1f}) mm  "
            f"RPY: ({rx:6.1f}, {ry:6.1f}, {rz:6.1f}) deg"
        )

    # ------------------------------------------------------------------
    # Route event handlers
    # ------------------------------------------------------------------

    def _on_route_changed(self):
        self.viewport.set_route(self.route)
        self.viewport.refresh()
        self._set_status(f"ルート更新 — {len(self.route)} 点")

    def _on_waypoint_selected(self, idx: int):
        self.viewport.set_selected_waypoint(idx)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _load_csv(self):
        path = filedialog.askopenfilename(
            title="CSV ファイルを開く",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            loaded = RouteCSVIO.route_from_csv(path)
            self.route.waypoints = loaded.waypoints
            self.route.name = loaded.name
            self.route.comment = loaded.comment
            self.route_editor.set_route(self.route)
            self.viewport.set_route(self.route)
            self.viewport.refresh()
            self._set_status(f"読込完了: {len(self.route)} 点 ← {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"CSV 読込失敗:\n{e}")

    def _save_csv(self):
        path = filedialog.asksaveasfilename(
            title="CSV として保存", defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{self.route.name}.csv")
        if not path:
            return
        try:
            RouteCSVIO.route_to_csv(self.route, path)
            self._set_status(f"保存完了: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"CSV 保存失敗:\n{e}")

    def _export_tp(self):
        if not self.route.waypoints:
            messagebox.showwarning("Warning", "ウェイポイントがありません。")
            return
        path = filedialog.asksaveasfilename(
            title="FANUC TP プログラムをエクスポート", defaultextension=".ls",
            filetypes=[("FANUC TP", "*.ls"), ("All files", "*.*")],
            initialfile=f"{self.route.name}.ls")
        if not path:
            return
        self._set_status("IK 計算中...")
        self.root.update()
        try:
            exporter = TPExporter(self.kin)
            exporter.export(self.route, path,
                            utool=self._active_tool.number,
                            uframe=self._active_uframe.number,
                            speed_override=self._speed_override.get())
            self._set_status(f"TP 出力完了: {os.path.basename(path)}")
            with open(path) as f:
                content = f.read()
            self._show_text_preview(f"TP: {os.path.basename(path)}", content)
        except Exception as e:
            messagebox.showerror("Error", f"TP エクスポート失敗:\n{e}")

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _start_simulation(self):
        if not self.route.waypoints:
            messagebox.showwarning("Warning", "ウェイポイントがありません。")
            return
        if self._sim_thread and self._sim_thread.is_alive():
            return
        self._sim_running = True
        self._sim_btn.config(state="disabled")
        self._set_status("シミュレーション実行中...")

        override = self._speed_override.get() / 100.0

        def run():
            q_prev = self._joint_angles.copy()
            waypoints = list(self.route.waypoints)
            for i, wp in enumerate(waypoints):
                if not self._sim_running:
                    break
                T = wp.to_transform()
                q_target, ok = self.kin.inverse(T, q_init=q_prev)
                if not ok:
                    self.root.after(0, lambda i=i: self._set_status(f"IK 失敗: P[{i+1}]"))
                    q_target = q_prev

                # Compute frames based on max joint speed and override
                speeds_rad = np.deg2rad(self.kin.dh.get_joint_max_speeds()) * override
                delta = np.abs(q_target - q_prev)
                max_time = float(np.max(delta / np.maximum(speeds_rad, 1e-6)))
                steps = max(20, int(max_time / 0.03))

                for step in range(steps + 1):
                    if not self._sim_running:
                        break
                    alpha = step / steps
                    q_interp = q_prev + alpha * (q_target - q_prev)

                    def _update(q=q_interp.copy(), idx=i):
                        self._joint_angles = q
                        self._update_viewport_from_angles(q)
                        self._update_fk_display()
                        self._update_angles_display()
                        for j, var in enumerate(self._slider_vars):
                            var.set(np.rad2deg(q[j]))
                        self.viewport.set_selected_waypoint(idx)
                        self._set_status(f"移動中: P[{idx+1}] / {len(waypoints)}  "
                                         f"({wp.label})")

                    self.root.after(0, _update)
                    import time
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
        self._set_status("シミュレーション完了")

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------

    def _compute_ik_for_wp(self):
        idx = self._ik_wp_var.get() - 1
        if idx < 0 or idx >= len(self.route.waypoints):
            messagebox.showwarning("Warning", f"P[{idx+1}] が存在しません。")
            return
        wp = self.route.waypoints[idx]
        T = wp.to_transform()
        self._set_status(f"IK 計算中: P[{idx+1}]...")
        self.root.update()
        q, ok = self.kin.inverse(T, q_init=self._joint_angles)
        if ok:
            self._set_angles(q)
            self.viewport.set_selected_waypoint(idx)
            self._set_status(f"IK 成功: P[{idx+1}]")
        else:
            messagebox.showwarning("IK 失敗",
                                   f"P[{idx+1}] の逆運動学計算に失敗しました。\n"
                                   f"位置: ({wp.x:.1f}, {wp.y:.1f}, {wp.z:.1f}) mm")

    # ------------------------------------------------------------------
    # Robot presets
    # ------------------------------------------------------------------

    def _go_home(self):
        self._set_angles(self.kin.dh.home_position())

    def _go_ready(self):
        self._set_angles(self.kin.dh.ready_position())

    def _set_angles(self, q: np.ndarray):
        self._joint_angles = q.copy()
        for i, var in enumerate(self._slider_vars):
            var.set(np.rad2deg(q[i]))
        self._update_viewport_from_angles(q)
        self._update_fk_display()
        self._update_angles_display()

    # ------------------------------------------------------------------
    # Tool / User frame dialogs
    # ------------------------------------------------------------------

    def _edit_tool_frame(self):
        tf = self._active_tool
        self._frame_editor_dialog(
            title=f"ツールフレーム編集: {tf.name}",
            obj=tf, fields=["x", "y", "z", "rx", "ry", "rz"],
            labels=["X (mm)", "Y (mm)", "Z (mm)", "Rx (°)", "Ry (°)", "Rz (°)"],
            on_apply=lambda: self.viewport.set_tool_frame(self._active_tool)
        )

    def _edit_user_frame(self):
        uf = self._active_uframe
        self._frame_editor_dialog(
            title=f"ユーザーフレーム編集: {uf.name}",
            obj=uf, fields=["x", "y", "z", "rx", "ry", "rz"],
            labels=["X (mm)", "Y (mm)", "Z (mm)", "Rx (°)", "Ry (°)", "Rz (°)"],
            on_apply=lambda: self.viewport.set_user_frame(self._active_uframe)
        )

    def _frame_editor_dialog(self, title, obj, fields, labels, on_apply):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("320x280")
        win.configure(bg="#1E1E1E")

        vars_ = {}
        for i, (f, lbl) in enumerate(zip(fields, labels)):
            row = ttk.Frame(win)
            row.pack(fill=tk.X, padx=12, pady=3)
            ttk.Label(row, text=lbl, width=8).pack(side=tk.LEFT)
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
            self._set_status(f"{title} 更新完了")
            win.destroy()

        btn_row = ttk.Frame(win)
        btn_row.pack(pady=8)
        ttk.Button(btn_row, text="適用 (Apply)", command=apply).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="キャンセル", command=win.destroy).pack(side=tk.LEFT, padx=4)

    # ------------------------------------------------------------------
    # Route sample
    # ------------------------------------------------------------------

    def _load_sample_route(self):
        sample = Route.default_sharpening_route()
        self.route.waypoints = sample.waypoints
        self.route.name = sample.name
        self.route_editor.set_route(self.route)
        self.viewport.set_route(self.route)
        self.viewport.refresh()
        self._set_status(f"サンプルルート読込: {len(self.route)} 点")

    def _auto_generate_route(self):
        """Open the auto-route generation dialog."""
        win = tk.Toplevel(self.root)
        win.title("刃付けルート自動生成")
        win.geometry("440x520")
        win.configure(bg="#1A1A1A")
        win.resizable(False, False)

        tk.Label(win, text="刃付けルート自動生成", bg="#1A1A1A", fg="#F5C400",
                 font=("", 12, "bold")).pack(pady=(12, 4))

        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)

        def row(label, default):
            f = ttk.Frame(frame)
            f.pack(fill=tk.X, pady=2)
            ttk.Label(f, text=label, width=30, anchor="w").pack(side=tk.LEFT)
            var = tk.StringVar(value=str(default))
            ttk.Entry(f, textvariable=var, width=10).pack(side=tk.LEFT)
            return var

        ttk.Label(frame, text="砥石位置 (ロボット基準座標)", foreground="#888888",
                  font=("", 8)).pack(anchor="w", pady=(6, 0))
        v_sx   = row("砥石 X mm (前方):", 400)
        v_sy   = row("砥石 Y mm (左右):", 0)
        v_sz   = row("砥石 Z mm (高さ):", 250)

        ttk.Label(frame, text="砥石寸法", foreground="#888888",
                  font=("", 8)).pack(anchor="w", pady=(6, 0))
        v_slen = row("砥石の長さ mm (ストローク方向):", 200)
        v_swid = row("砥石の幅  mm (包丁送り方向):", 70)

        ttk.Label(frame, text="刃付けパラメータ", foreground="#888888",
                  font=("", 8)).pack(anchor="w", pady=(6, 0))
        v_ang  = row("刃角度 deg:", 15)
        v_blen = row("研磨刃長 mm:", 180)
        v_strk = row("往復ストローク回数:", 5)
        v_spd  = row("ストローク速度 mm/s:", 30)

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
                self.route.name = new_route.name
                self.route_editor.set_route(self.route)
                self.viewport.set_route(self.route)
                self.viewport.refresh()
                self._set_status(f"ルート自動生成完了: {len(self.route)} 点")
                win.destroy()
            except Exception as e:
                messagebox.showerror("エラー", str(e), parent=win)

        ttk.Button(win, text="ルートを生成", command=on_generate).pack(pady=6)
        ttk.Button(win, text="キャンセル", command=win.destroy).pack(pady=2)

    def _clear_route(self):
        if messagebox.askyesno("確認", "ルートをすべてクリアしますか？"):
            self.route.clear()
            self.route_editor.set_route(self.route)
            self.viewport.set_route(self.route)
            self.viewport.refresh()
            self._set_status("ルートをクリアしました")

    # ------------------------------------------------------------------
    # Info dialogs
    # ------------------------------------------------------------------

    def _show_dh_params(self):
        self._show_text_preview("DH パラメータ", repr(self.kin.dh))

    def _show_robot_specs(self):
        dh = self.kin.dh
        specs = (
            f"FANUC LR Mate 200iD/14L  ロボット仕様\n"
            f"{'='*50}\n"
            f"ペイロード      : {dh.PAYLOAD_KG} kg\n"
            f"最大リーチ      : {dh.REACH_MM} mm (手首中心まで)\n"
            f"繰り返し精度    : ±{dh.REPEATABILITY_MM} mm\n"
            f"ロボット質量    : {dh.WEIGHT_KG} kg\n"
            f"コントローラ    : {dh.CONTROLLER}\n"
            f"防塵防水        : {dh.IP_RATING}\n"
            f"\n{'='*50}\n"
            f"{'軸':4} {'最小(°)':>10} {'最大(°)':>10} {'最大速度(°/s)':>14}\n"
            f"{'-'*40}\n"
        )
        for j in dh.joints:
            specs += f"{j.name:4} {j.joint_min:>10.0f} {j.joint_max:>10.0f} {j.joint_max_speed:>14.0f}\n"
        self._show_text_preview("ロボット仕様", specs)

    def _show_about(self):
        msg = (
            f"FANUC LR Mate 200iD/14L\n"
            f"刃付けロボットシミュレータ  v{APP_VERSION}\n\n"
            f"Knife Sharpening Robot Simulator\n\n"
            f"Tech: Python · matplotlib · tkinter\n"
            f"Kinematics: Modified DH (6-DOF, Z-up)\n"
            f"IK: Analytical + Numerical fallback\n"
        )
        messagebox.showinfo("About", msg)

    def _show_text_preview(self, title: str, content: str):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("700x500")
        txt = scrolledtext.ScrolledText(
            win, font=("Courier", 9), bg="#1A1A1A", fg="#DDDDDD",
            insertbackground="white", wrap=tk.NONE)
        txt.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        txt.insert(tk.END, content)
        txt.config(state="disabled")
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=4)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _set_status(self, msg: str):
        self._status_var.set(f"{msg}   [v{APP_VERSION}]")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_close(self):
        self._sim_running = False
        if self._simulator is not None:
            try:
                self._simulator.stop()
            except Exception:
                pass
        self.viewport.destroy()
        self.root.destroy()

    def run(self):
        self.root.mainloop()
