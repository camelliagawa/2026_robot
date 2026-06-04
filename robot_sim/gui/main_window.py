"""
Main tkinter window for the FANUC LR Mate 200iD/14L knife sharpening simulator.

Layout:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Menu bar                                                        │
  ├──────────────────────────────┬──────────────────────────────────┤
  │  3D Viewport (matplotlib)    │  Route Editor (waypoint list)    │
  │                              │                                  │
  │                              │  [ Add ] [Edit] [Del] [↑] [↓]   │
  ├──────────────────────────────┴──────────────────────────────────┤
  │  Joint sliders (J1-J6)                                          │
  ├─────────────────────────────────────────────────────────────────┤
  │  Controls: [Load CSV] [Save CSV] [Simulate] [Export TP] [FK/IK] │
  └─────────────────────────────────────────────────────────────────┘
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
from ..path.route import Route, Waypoint, MotionType
from ..path.csv_io import RouteCSVIO
from ..path.tp_exporter import TPExporter
from .viewport import Viewport3D
from .route_editor import RouteEditor


class MainWindow:
    """
    Top-level application window.

    Instantiate and call .run() to start the GUI event loop.
    """

    APP_TITLE = "FANUC LR Mate 200iD/14L  |  刃付けロボットシミュレータ"
    MIN_WIDTH = 1200
    MIN_HEIGHT = 780

    def __init__(self):
        self.kin = Kinematics()
        self.route = Route.default_sharpening_route()
        self._joint_angles = self.kin.dh.ready_position().copy()
        self._sim_thread: Optional[threading.Thread] = None

        # Simulator (lazy init only when requested)
        self._simulator = None

        self._build_root()
        self._build_menu()
        self._build_main_panels()
        self._build_joint_sliders()
        self._build_bottom_controls()
        self._build_status_bar()

        # Initial viewport update
        self.viewport.set_route(self.route)
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

        # Apply dark theme
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

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu(self):
        menubar = tk.Menu(self.root, bg="#1E1E1E", fg="#DDDDDD",
                          activebackground="#4A4A6A", activeforeground="white")

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0, bg="#2A2A2A", fg="#DDDDDD",
                            activebackground="#4A4A6A")
        file_menu.add_command(label="CSVを開く (Open CSV)...", command=self._load_csv,
                              accelerator="Ctrl+O")
        file_menu.add_command(label="CSVとして保存 (Save CSV)...", command=self._save_csv,
                              accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="TPプログラムをエクスポート...", command=self._export_tp,
                              accelerator="Ctrl+E")
        file_menu.add_separator()
        file_menu.add_command(label="終了 (Exit)", command=self._on_close)
        menubar.add_cascade(label="ファイル (File)", menu=file_menu)

        # Route menu
        route_menu = tk.Menu(menubar, tearoff=0, bg="#2A2A2A", fg="#DDDDDD",
                             activebackground="#4A4A6A")
        route_menu.add_command(label="サンプルルートを読み込む", command=self._load_sample_route)
        route_menu.add_command(label="ルートをクリア", command=self._clear_route)
        route_menu.add_separator()
        route_menu.add_command(label="シミュレーション実行", command=self._start_simulation,
                               accelerator="F5")
        menubar.add_cascade(label="ルート (Route)", menu=route_menu)

        # Robot menu
        robot_menu = tk.Menu(menubar, tearoff=0, bg="#2A2A2A", fg="#DDDDDD",
                              activebackground="#4A4A6A")
        robot_menu.add_command(label="ホームポジション", command=self._go_home)
        robot_menu.add_command(label="レディポジション", command=self._go_ready)
        robot_menu.add_separator()
        robot_menu.add_command(label="DH パラメータを表示", command=self._show_dh_params)
        menubar.add_cascade(label="ロボット (Robot)", menu=robot_menu)

        # Help
        help_menu = tk.Menu(menubar, tearoff=0, bg="#2A2A2A", fg="#DDDDDD",
                            activebackground="#4A4A6A")
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="ヘルプ (Help)", menu=help_menu)

        self.root.config(menu=menubar)
        self.root.bind("<Control-o>", lambda e: self._load_csv())
        self.root.bind("<Control-s>", lambda e: self._save_csv())
        self.root.bind("<Control-e>", lambda e: self._export_tp())
        self.root.bind("<F5>", lambda e: self._start_simulation())

    # ------------------------------------------------------------------
    # Main panels (viewport + route editor)
    # ------------------------------------------------------------------

    def _build_main_panels(self):
        # Main paned window
        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left: 3D viewport
        left_frame = ttk.LabelFrame(pane, text="3D ビューポート (3D Viewport)")
        pane.add(left_frame, weight=3)

        self.viewport = Viewport3D(left_frame, self.kin)

        # Right: Route editor
        right_frame = ttk.LabelFrame(pane, text="ルートエディタ (Route Editor)")
        pane.add(right_frame, weight=1)

        self.route_editor = RouteEditor(
            right_frame,
            self.route,
            on_change=self._on_route_changed,
            on_select=self._on_waypoint_selected,
        )
        self.route_editor.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ------------------------------------------------------------------
    # Joint sliders
    # ------------------------------------------------------------------

    def _build_joint_sliders(self):
        """Build J1–J6 joint angle sliders."""
        slider_frame = ttk.LabelFrame(
            self.root, text="ジョイント角度 (Joint Angles)"
        )
        slider_frame.pack(fill=tk.X, padx=4, pady=2)

        lower, upper = self.kin.dh.get_joint_limits_deg()
        self._slider_vars = []

        for i in range(6):
            col_frame = ttk.Frame(slider_frame)
            col_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4, pady=2)

            init_deg = np.rad2deg(self._joint_angles[i])
            var = tk.DoubleVar(value=init_deg)
            self._slider_vars.append(var)

            lbl = ttk.Label(col_frame, text=f"J{i+1}", width=4, anchor="center")
            lbl.pack()

            slider = ttk.Scale(
                col_frame,
                from_=lower[i], to=upper[i],
                variable=var,
                orient=tk.VERTICAL,
                length=100,
                command=lambda val, idx=i: self._on_slider_change(idx, float(val)),
            )
            slider.pack()

            val_lbl = ttk.Label(col_frame, textvariable=var, width=6, anchor="center")
            val_lbl.pack()

        # Value display (formatted)
        self._angles_display_var = tk.StringVar()
        ttk.Label(slider_frame, textvariable=self._angles_display_var,
                  font=("Courier", 8), foreground="#888888").pack(
            side=tk.RIGHT, padx=8
        )
        self._update_angles_display()

    def _on_slider_change(self, joint_idx: int, value_deg: float):
        """Handle slider drag."""
        self._joint_angles[joint_idx] = np.deg2rad(value_deg)
        self._update_viewport_from_angles(self._joint_angles)
        self._update_fk_display()
        self._update_angles_display()

    def _update_angles_display(self):
        deg = np.rad2deg(self._joint_angles)
        self._angles_display_var.set(
            "  ".join(f"J{i+1}:{d:+7.1f}°" for i, d in enumerate(deg))
        )

    # ------------------------------------------------------------------
    # Bottom controls bar
    # ------------------------------------------------------------------

    def _build_bottom_controls(self):
        ctrl_frame = ttk.Frame(self.root)
        ctrl_frame.pack(fill=tk.X, padx=4, pady=4)

        # File operations
        io_frame = ttk.LabelFrame(ctrl_frame, text="ファイル I/O")
        io_frame.pack(side=tk.LEFT, padx=4)
        ttk.Button(io_frame, text="CSV 読込", command=self._load_csv, width=10).pack(
            side=tk.LEFT, padx=2, pady=2
        )
        ttk.Button(io_frame, text="CSV 保存", command=self._save_csv, width=10).pack(
            side=tk.LEFT, padx=2, pady=2
        )
        ttk.Button(io_frame, text="TP エクスポート", command=self._export_tp, width=12).pack(
            side=tk.LEFT, padx=2, pady=2
        )

        # Simulation
        sim_frame = ttk.LabelFrame(ctrl_frame, text="シミュレーション")
        sim_frame.pack(side=tk.LEFT, padx=4)
        self._sim_btn = ttk.Button(
            sim_frame, text="▶ シミュレーション実行", command=self._start_simulation, width=18
        )
        self._sim_btn.pack(side=tk.LEFT, padx=2, pady=2)
        ttk.Button(sim_frame, text="■ 停止", command=self._stop_simulation, width=8).pack(
            side=tk.LEFT, padx=2, pady=2
        )

        # IK section
        ik_frame = ttk.LabelFrame(ctrl_frame, text="逆運動学 (IK)")
        ik_frame.pack(side=tk.LEFT, padx=4)
        ttk.Label(ik_frame, text="WP→IK:").pack(side=tk.LEFT)
        self._ik_wp_var = tk.IntVar(value=1)
        ttk.Spinbox(
            ik_frame, from_=1, to=999, textvariable=self._ik_wp_var,
            width=4
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(ik_frame, text="IK 計算", command=self._compute_ik_for_wp, width=8).pack(
            side=tk.LEFT, padx=2, pady=2
        )

        # FK output display
        fk_frame = ttk.LabelFrame(ctrl_frame, text="FK 結果 (mm / deg)")
        fk_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self._fk_display_var = tk.StringVar()
        ttk.Label(
            fk_frame, textvariable=self._fk_display_var,
            font=("Courier", 8), foreground="#88CCFF"
        ).pack(anchor="w", padx=4)

    def _build_status_bar(self):
        self._status_var = tk.StringVar(value="Ready  /  準備完了")
        status_bar = ttk.Label(
            self.root, textvariable=self._status_var,
            relief=tk.SUNKEN, anchor=tk.W, font=("", 8)
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=2, pady=1)

    # ------------------------------------------------------------------
    # Viewport & FK helpers
    # ------------------------------------------------------------------

    def _update_viewport_from_angles(self, q: np.ndarray):
        """Push joint angles to viewport."""
        self.viewport.update_robot(q)

    def _update_fk_display(self):
        """Compute FK and update display label."""
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
        """Route was modified — refresh viewport."""
        self.viewport.set_route(self.route)
        self.viewport.refresh()
        n = len(self.route)
        self._set_status(f"Route updated — {n} waypoints")

    def _on_waypoint_selected(self, idx: int):
        """A waypoint was selected in the editor — highlight it."""
        self.viewport.set_selected_waypoint(idx)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _load_csv(self):
        path = filedialog.askopenfilename(
            title="CSV ファイルを開く",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            loaded_route = RouteCSVIO.route_from_csv(path)
            # Replace route waypoints
            self.route.waypoints = loaded_route.waypoints
            self.route.name = loaded_route.name
            self.route.comment = loaded_route.comment
            self.route_editor.set_route(self.route)
            self.viewport.set_route(self.route)
            self.viewport.refresh()
            self._set_status(f"Loaded {len(self.route)} waypoints from {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n{e}")

    def _save_csv(self):
        path = filedialog.asksaveasfilename(
            title="CSV として保存",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{self.route.name}.csv",
        )
        if not path:
            return
        try:
            RouteCSVIO.route_to_csv(self.route, path)
            self._set_status(f"Saved to {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save CSV:\n{e}")

    def _export_tp(self):
        if not self.route.waypoints:
            messagebox.showwarning("Warning", "ルートにウェイポイントがありません。")
            return

        path = filedialog.asksaveasfilename(
            title="FANUC TP プログラムをエクスポート",
            defaultextension=".ls",
            filetypes=[("FANUC TP files", "*.ls"), ("All files", "*.*")],
            initialfile=f"{self.route.name}.ls",
        )
        if not path:
            return

        self._set_status("IK 計算中... (Computing IK for TP export)")
        self.root.update()

        try:
            exporter = TPExporter(self.kin)
            exporter.export(self.route, path)
            self._set_status(f"TP exported to {os.path.basename(path)}")

            # Show preview in popup
            with open(path, "r") as f:
                content = f.read()
            self._show_text_preview(f"TP Program: {os.path.basename(path)}", content)

        except Exception as e:
            messagebox.showerror("Error", f"TP エクスポートに失敗しました:\n{e}")

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _start_simulation(self):
        """Animate the robot through the route waypoints (GUI only, no PyBullet)."""
        if not self.route.waypoints:
            messagebox.showwarning("Warning", "ルートにウェイポイントがありません。")
            return

        if self._sim_thread and self._sim_thread.is_alive():
            return  # already running

        self._sim_running = True
        self._sim_btn.config(state="disabled")
        self._set_status("シミュレーション実行中... (Simulation running)")

        def run():
            q_prev = self._joint_angles.copy()
            waypoints = list(self.route.waypoints)
            steps = 40  # frames per segment

            for i, wp in enumerate(waypoints):
                if not self._sim_running:
                    break

                T = wp.to_transform()
                q_target, ok = self.kin.inverse(T, q_init=q_prev)
                if not ok:
                    self.root.after(0, lambda i=i: self._set_status(
                        f"IK failed for P[{i+1}]  /  P[{i+1}] の IK に失敗"
                    ))
                    q_target = q_prev

                # Interpolate
                for step in range(steps + 1):
                    if not self._sim_running:
                        break
                    alpha = step / steps
                    q_interp = q_prev + alpha * (q_target - q_prev)

                    # Update GUI from simulation thread safely
                    def _update(q=q_interp.copy(), idx=i):
                        self._joint_angles = q
                        self._update_viewport_from_angles(q)
                        self._update_fk_display()
                        self._update_angles_display()
                        for j, var in enumerate(self._slider_vars):
                            var.set(np.rad2deg(q[j]))
                        self.viewport.set_selected_waypoint(idx)
                        self._set_status(
                            f"Moving to P[{idx+1}]  ({idx+1}/{len(waypoints)})"
                        )

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
        self._set_status("シミュレーション完了 (Simulation complete)")

    # ------------------------------------------------------------------
    # IK computation
    # ------------------------------------------------------------------

    def _compute_ik_for_wp(self):
        """Compute IK for a selected waypoint and move sliders."""
        idx = self._ik_wp_var.get() - 1
        if idx < 0 or idx >= len(self.route.waypoints):
            messagebox.showwarning("Warning", f"Waypoint P[{idx+1}] not found.")
            return

        wp = self.route.waypoints[idx]
        T = wp.to_transform()
        self._set_status(f"Computing IK for P[{idx+1}]...")
        self.root.update()

        q, ok = self.kin.inverse(T, q_init=self._joint_angles)
        if ok:
            self._joint_angles = q
            for i, var in enumerate(self._slider_vars):
                var.set(np.rad2deg(q[i]))
            self._update_viewport_from_angles(q)
            self._update_fk_display()
            self._update_angles_display()
            self.viewport.set_selected_waypoint(idx)
            self._set_status(f"IK success for P[{idx+1}]  /  IK 成功")
        else:
            messagebox.showwarning(
                "IK Failed",
                f"P[{idx+1}] の逆運動学計算に失敗しました。\n"
                f"Position: ({wp.x:.1f}, {wp.y:.1f}, {wp.z:.1f}) mm"
            )

    # ------------------------------------------------------------------
    # Robot presets
    # ------------------------------------------------------------------

    def _go_home(self):
        q = self.kin.dh.home_position()
        self._set_angles(q)

    def _go_ready(self):
        q = self.kin.dh.ready_position()
        self._set_angles(q)

    def _set_angles(self, q: np.ndarray):
        self._joint_angles = q.copy()
        for i, var in enumerate(self._slider_vars):
            var.set(np.rad2deg(q[i]))
        self._update_viewport_from_angles(q)
        self._update_fk_display()
        self._update_angles_display()

    # ------------------------------------------------------------------
    # Sample route
    # ------------------------------------------------------------------

    def _load_sample_route(self):
        sample = Route.default_sharpening_route()
        self.route.waypoints = sample.waypoints
        self.route.name = sample.name
        self.route.comment = sample.comment
        self.route_editor.set_route(self.route)
        self.viewport.set_route(self.route)
        self.viewport.refresh()
        self._set_status(f"サンプルルートを読み込みました ({len(self.route)} 点)")

    def _clear_route(self):
        from tkinter import messagebox
        if messagebox.askyesno("確認", "ルートをすべてクリアしますか？"):
            self.route.clear()
            self.route_editor.set_route(self.route)
            self.viewport.set_route(self.route)
            self.viewport.refresh()
            self._set_status("Route cleared")

    # ------------------------------------------------------------------
    # Info dialogs
    # ------------------------------------------------------------------

    def _show_dh_params(self):
        text = repr(self.kin.dh)
        self._show_text_preview("DH Parameters — FANUC LR Mate 200iD/14L", text)

    def _show_about(self):
        msg = (
            "FANUC LR Mate 200iD/14L\n"
            "刃付けロボットシミュレータ\n\n"
            "Knife Sharpening Robot Simulator\n\n"
            "Tech: Python · PyBullet · matplotlib · tkinter\n"
            "Kinematics: Modified DH (6-DOF)\n"
            "IK: Analytical + Numerical fallback\n"
        )
        messagebox.showinfo("About", msg)

    def _show_text_preview(self, title: str, content: str):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("700x500")

        text = scrolledtext.ScrolledText(
            win, font=("Courier", 9), bg="#1A1A1A", fg="#DDDDDD",
            insertbackground="white", wrap=tk.NONE
        )
        text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        text.insert(tk.END, content)
        text.config(state="disabled")

        ttk.Button(win, text="Close", command=win.destroy).pack(pady=4)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _set_status(self, msg: str):
        self._status_var.set(msg)

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
        """Start the tkinter main loop."""
        self.root.mainloop()
