"""
Route Editor Panel — tkinter widget for managing waypoints.

Shows a list of waypoints with add/remove/edit capabilities.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from typing import Optional, Callable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..path.route import Route, Waypoint

from ..path.route import Waypoint, Route, MotionType


class WaypointDialog(tk.Toplevel):
    """Dialog for creating or editing a single waypoint."""

    FIELDS = [
        ("X (mm)",   "x"),
        ("Y (mm)",   "y"),
        ("Z (mm)",   "z"),
        ("RX (deg)", "rx"),
        ("RY (deg)", "ry"),
        ("RZ (deg)", "rz"),
        ("Speed (mm/s)", "speed"),
        ("Label",    "label"),
    ]

    def __init__(self, parent, waypoint: Optional[Waypoint] = None, title: str = "Waypoint"):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.result: Optional[Waypoint] = None

        wp = waypoint or Waypoint()
        self._vars = {}

        frame = ttk.Frame(self, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # Fields
        for row, (label, attr) in enumerate(self.FIELDS):
            ttk.Label(frame, text=label, width=14, anchor="w").grid(
                row=row, column=0, sticky="w", pady=2
            )
            val = getattr(wp, attr)
            var = tk.StringVar(value=str(val))
            self._vars[attr] = var
            entry = ttk.Entry(frame, textvariable=var, width=16)
            entry.grid(row=row, column=1, sticky="ew", pady=2)
            if row == 0:
                entry.focus()

        # Motion type selector
        row = len(self.FIELDS)
        ttk.Label(frame, text="Motion Type", width=14, anchor="w").grid(
            row=row, column=0, sticky="w", pady=2
        )
        self._mt_var = tk.StringVar(value=wp.motion_type.value)
        mt_combo = ttk.Combobox(
            frame, textvariable=self._mt_var,
            values=["L", "J", "C"], state="readonly", width=14
        )
        mt_combo.grid(row=row, column=1, sticky="ew", pady=2)

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row + 1, column=0, columnspan=2, pady=(12, 0))
        ttk.Button(btn_frame, text="OK", command=self._ok, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy, width=10).pack(side=tk.LEFT, padx=4)

        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self.destroy())
        self.grab_set()
        self.transient(parent)
        self.wait_window()

    def _ok(self):
        try:
            wp = Waypoint(
                x=float(self._vars["x"].get()),
                y=float(self._vars["y"].get()),
                z=float(self._vars["z"].get()),
                rx=float(self._vars["rx"].get()),
                ry=float(self._vars["ry"].get()),
                rz=float(self._vars["rz"].get()),
                speed=float(self._vars["speed"].get()),
                label=self._vars["label"].get().strip(),
                motion_type=MotionType(self._mt_var.get()),
            )
            self.result = wp
            self.destroy()
        except ValueError as e:
            messagebox.showerror("Invalid input", f"Please enter valid numbers.\n{e}", parent=self)


class RouteEditor(ttk.Frame):
    """
    Panel widget for editing the route waypoints.

    Provides:
      - Listbox showing all waypoints
      - Add / Edit / Delete / Move Up / Move Down buttons
      - Callback hooks for viewport synchronization
    """

    def __init__(
        self,
        parent: tk.Widget,
        route: Route,
        on_change: Optional[Callable] = None,
        on_select: Optional[Callable[[int], None]] = None,
        listbox_height: int = 20,
        **kwargs,
    ):
        """
        Args:
            parent    : Parent widget.
            route     : The Route being edited (mutable).
            on_change : Called after any route modification.
            on_select : Called with waypoint index when selection changes.
        """
        super().__init__(parent, **kwargs)
        self.route = route
        self.on_change = on_change
        self.on_select = on_select
        self._listbox_height = listbox_height

        self._build_ui()
        self._refresh_list()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Build the editor panel layout."""
        # Title
        ttk.Label(self, text="Waypoints (経路点)", font=("", 11, "bold")).pack(
            pady=(6, 2)
        )

        # Route stats
        self._stats_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._stats_var, foreground="#888888",
                  font=("", 8)).pack()

        # Listbox with scrollbar
        list_frame = ttk.Frame(self)
        list_frame.pack(fill=tk.X, padx=4, pady=4)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE,
            font=("Courier", 9),
            bg="#2A2A2A",
            fg="#DDDDDD",
            selectbackground="#2255AA",
            selectforeground="white",
            activestyle="none",
            height=self._listbox_height,
        )
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self._listbox.yview)
        self._listbox.bind("<<ListboxSelect>>", self._on_listbox_select)
        self._listbox.bind("<Double-Button-1>", lambda e: self._edit_selected())

        # Button row
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=4, pady=2)

        buttons = [
            ("追加 (Add)",   self._add_waypoint),
            ("編集 (Edit)",  self._edit_selected),
            ("削除 (Del)",   self._delete_selected),
            ("↑ Up",        self._move_up),
            ("↓ Down",      self._move_down),
        ]
        for text, cmd in buttons:
            ttk.Button(btn_frame, text=text, command=cmd, width=10).pack(
                side=tk.LEFT, padx=1, pady=2
            )

        # Second button row
        btn_frame2 = ttk.Frame(self)
        btn_frame2.pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(btn_frame2, text="クリア (Clear)", command=self._clear_route, width=14).pack(
            side=tk.LEFT, padx=1
        )
        ttk.Button(btn_frame2, text="サンプル (Sample)", command=self._load_sample, width=14).pack(
            side=tk.LEFT, padx=1
        )

        # Waypoint detail view
        detail_frame = ttk.LabelFrame(self, text="Selected Waypoint Details")
        detail_frame.pack(fill=tk.X, padx=4, pady=4)
        self._detail_var = tk.StringVar(value="(no selection)")
        ttk.Label(
            detail_frame, textvariable=self._detail_var,
            font=("Courier", 8), justify=tk.LEFT, foreground="#AAAAAA"
        ).pack(anchor="w", padx=4, pady=4)

    # ------------------------------------------------------------------
    # List management
    # ------------------------------------------------------------------

    def _refresh_list(self):
        """Rebuild the listbox from the route."""
        sel = self._get_selection()
        self._listbox.delete(0, tk.END)

        for i, wp in enumerate(self.route.waypoints):
            label = wp.label or f"P[{i+1}]"
            mt = wp.motion_type.value
            line = (
                f"{i+1:3d} [{mt}] ({wp.x:7.1f},{wp.y:7.1f},{wp.z:7.1f})"
                f"  {wp.speed:5.0f}mm/s  {label}"
            )
            self._listbox.insert(tk.END, line)

        # Restore selection
        if sel is not None and sel < len(self.route.waypoints):
            self._listbox.selection_set(sel)
            self._listbox.see(sel)

        # Update stats
        n = len(self.route.waypoints)
        length = self.route.total_length_mm()
        t = self.route.estimated_time_sec()
        self._stats_var.set(
            f"{n} points  |  {length:.0f} mm  |  ~{t:.1f} s"
        )

    def _get_selection(self) -> Optional[int]:
        """Return currently selected listbox index, or None."""
        sel = self._listbox.curselection()
        return sel[0] if sel else None

    def _on_listbox_select(self, event=None):
        """Handle listbox selection change."""
        idx = self._get_selection()
        if idx is not None and idx < len(self.route.waypoints):
            wp = self.route.waypoints[idx]
            self._detail_var.set(
                f"Pos: ({wp.x:.2f}, {wp.y:.2f}, {wp.z:.2f}) mm\n"
                f"RPY: ({wp.rx:.2f}, {wp.ry:.2f}, {wp.rz:.2f}) deg\n"
                f"Speed: {wp.speed:.1f} mm/s  |  Type: {wp.motion_type.value}\n"
                f"Label: {wp.label or '(none)'}"
            )
            if self.on_select:
                self.on_select(idx)
        else:
            self._detail_var.set("(no selection)")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _add_waypoint(self):
        """Open dialog to add a new waypoint."""
        # Default to last waypoint position + small offset
        template = None
        if self.route.waypoints:
            last = self.route.waypoints[-1]
            template = Waypoint(
                x=last.x + 10, y=last.y, z=last.z,
                rx=last.rx, ry=last.ry, rz=last.rz,
                speed=last.speed, motion_type=last.motion_type
            )
        dlg = WaypointDialog(self.winfo_toplevel(), template, title="Add Waypoint")
        if dlg.result is not None:
            self.route.add_waypoint(dlg.result)
            self._notify_change()

    def _edit_selected(self):
        """Edit the selected waypoint."""
        idx = self._get_selection()
        if idx is None:
            messagebox.showinfo("Info", "Please select a waypoint to edit.")
            return
        wp = self.route.waypoints[idx]
        dlg = WaypointDialog(self.winfo_toplevel(), wp, title=f"Edit Waypoint P[{idx+1}]")
        if dlg.result is not None:
            # Preserve original ID
            dlg.result.id = wp.id
            self.route.waypoints[idx] = dlg.result
            self._notify_change()

    def _delete_selected(self):
        """Delete the selected waypoint."""
        idx = self._get_selection()
        if idx is None:
            messagebox.showinfo("Info", "Please select a waypoint to delete.")
            return
        wp = self.route.waypoints[idx]
        label = wp.label or f"P[{idx+1}]"
        if messagebox.askyesno("Confirm", f"Delete waypoint '{label}'?"):
            self.route.remove_waypoint(idx)
            self._notify_change()

    def _move_up(self):
        idx = self._get_selection()
        if idx is not None and idx > 0:
            self.route.move_waypoint(idx, idx - 1)
            self._listbox.selection_clear(0, tk.END)
            self._listbox.selection_set(idx - 1)
            self._notify_change()

    def _move_down(self):
        idx = self._get_selection()
        if idx is not None and idx < len(self.route.waypoints) - 1:
            self.route.move_waypoint(idx, idx + 1)
            self._listbox.selection_clear(0, tk.END)
            self._listbox.selection_set(idx + 1)
            self._notify_change()

    def _clear_route(self):
        if self.route.waypoints:
            if messagebox.askyesno("Confirm", "Clear all waypoints?"):
                self.route.clear()
                self._notify_change()

    def _load_sample(self):
        """Load the default sharpening demo route."""
        if self.route.waypoints:
            if not messagebox.askyesno("Confirm", "Replace current route with sample?"):
                return
        sample = Route.default_sharpening_route()
        self.route.waypoints = sample.waypoints
        self.route.name = sample.name
        self.route.comment = sample.comment
        self._notify_change()

    # ------------------------------------------------------------------
    # Notification
    # ------------------------------------------------------------------

    def _notify_change(self):
        """Refresh list and call on_change callback."""
        self._refresh_list()
        if self.on_change:
            self.on_change()

    def set_route(self, route: Route):
        """Replace the route being edited."""
        self.route = route
        self._refresh_list()
