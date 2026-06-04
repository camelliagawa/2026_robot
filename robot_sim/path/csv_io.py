"""
CSV import/export for robot routes.

CSV format:
    # x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg, speed_mmps, motion_type[, label]
    300, 0, 400, 180, 0, 0, 50, L, Home
    350, 50, 380, 180, 0, 0, 50, L
"""
from __future__ import annotations

import csv
import io
import os
from typing import List, Optional, TextIO, Union

from .route import Waypoint, Route, MotionType


# Column definitions
CSV_FIELDNAMES = ["x_mm", "y_mm", "z_mm", "rx_deg", "ry_deg", "rz_deg",
                  "speed_mmps", "motion_type", "label"]


class RouteCSVIO:
    """
    Read and write Route / Waypoint lists as CSV files.
    """

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @staticmethod
    def route_to_csv(route: Route, file_path: str, write_header: bool = True):
        """
        Save a Route to a CSV file.

        Args:
            route      : Route object to export.
            file_path  : Destination file path.
            write_header: Write comment/header lines at top.
        """
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            RouteCSVIO.route_to_file(route, f, write_header=write_header)

    @staticmethod
    def route_to_file(route: Route, f: TextIO, write_header: bool = True):
        """Write route to any file-like object."""
        if write_header:
            f.write(f"# Route: {route.name}\n")
            f.write(f"# Comment: {route.comment}\n")
            f.write(f"# UFrame: {route.uframe}, UTool: {route.utool}\n")
            f.write(f"# Columns: {', '.join(CSV_FIELDNAMES)}\n")

        writer = csv.writer(f)
        writer.writerow(CSV_FIELDNAMES)  # header row

        for wp in route.waypoints:
            writer.writerow([
                f"{wp.x:.4f}",
                f"{wp.y:.4f}",
                f"{wp.z:.4f}",
                f"{wp.rx:.4f}",
                f"{wp.ry:.4f}",
                f"{wp.rz:.4f}",
                f"{wp.speed:.4f}",
                wp.motion_type.value,
                wp.label,
            ])

    @staticmethod
    def route_to_string(route: Route) -> str:
        """Return CSV content as string."""
        buf = io.StringIO()
        RouteCSVIO.route_to_file(route, buf)
        return buf.getvalue()

    @staticmethod
    def waypoints_to_csv(waypoints: List[Waypoint], file_path: str):
        """Save just a list of waypoints (no route metadata)."""
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_FIELDNAMES)
            for wp in waypoints:
                writer.writerow([
                    wp.x, wp.y, wp.z,
                    wp.rx, wp.ry, wp.rz,
                    wp.speed, wp.motion_type.value, wp.label
                ])

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    @staticmethod
    def route_from_csv(file_path: str) -> Route:
        """
        Load a Route from a CSV file.

        Parses comment lines beginning with '#' for route metadata.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return RouteCSVIO.route_from_string(content)

    @staticmethod
    def route_from_string(content: str) -> Route:
        """Parse CSV content string into a Route."""
        route = Route()

        # Parse metadata from comment lines
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("# Route:"):
                route.name = stripped[len("# Route:"):].strip()
            elif stripped.startswith("# Comment:"):
                route.comment = stripped[len("# Comment:"):].strip()
            elif stripped.startswith("# UFrame:"):
                parts = stripped[len("# UFrame:"):].strip().split(",")
                try:
                    route.uframe = int(parts[0].strip())
                    if len(parts) > 1 and "UTool:" in parts[1]:
                        route.utool = int(parts[1].split(":")[1].strip())
                except ValueError:
                    pass

        # Parse data rows
        reader = csv.DictReader(
            line for line in content.splitlines()
            if not line.strip().startswith("#")
        )

        for row in reader:
            try:
                wp = RouteCSVIO._row_to_waypoint(row)
                route.add_waypoint(wp)
            except (KeyError, ValueError) as e:
                print(f"[CSV] Skipping invalid row {row}: {e}")

        return route

    @staticmethod
    def waypoints_from_csv(file_path: str) -> List[Waypoint]:
        """Load a list of waypoints from CSV (no route metadata needed)."""
        route = RouteCSVIO.route_from_csv(file_path)
        return route.waypoints

    @staticmethod
    def waypoints_from_string(content: str) -> List[Waypoint]:
        """Parse waypoints from CSV string."""
        return RouteCSVIO.route_from_string(content).waypoints

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_waypoint(row: dict) -> Waypoint:
        """Convert a CSV row dict to a Waypoint."""
        # Support both named and positional columns
        def get(key, alt_keys=None, default=0.0):
            for k in [key] + (alt_keys or []):
                if k in row and row[k] not in (None, ""):
                    return row[k]
            return default

        # Handle column aliases
        x = float(get("x_mm", ["x"], "0"))
        y = float(get("y_mm", ["y"], "0"))
        z = float(get("z_mm", ["z"], "400"))
        rx = float(get("rx_deg", ["rx"], "180"))
        ry = float(get("ry_deg", ["ry"], "0"))
        rz = float(get("rz_deg", ["rz"], "0"))
        speed = float(get("speed_mmps", ["speed"], "50"))
        mt_str = str(get("motion_type", [], "L")).strip().upper()
        label = str(get("label", [], "")).strip()

        # Normalize motion type
        if mt_str in ("L", "LINEAR"):
            mt = MotionType.LINEAR
        elif mt_str in ("J", "JOINT"):
            mt = MotionType.JOINT
        elif mt_str in ("C", "CIRCULAR"):
            mt = MotionType.CIRCULAR
        else:
            mt = MotionType.LINEAR

        return Waypoint(
            x=x, y=y, z=z,
            rx=rx, ry=ry, rz=rz,
            speed=speed,
            motion_type=mt,
            label=label,
        )

    @staticmethod
    def example_csv_content() -> str:
        """Return example CSV content string."""
        return (
            "# x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg, speed_mmps, motion_type\n"
            "x_mm,y_mm,z_mm,rx_deg,ry_deg,rz_deg,speed_mmps,motion_type,label\n"
            "300,0,450,180,0,0,30,J,Home\n"
            "300,-100,300,180,15,0,50,L,Approach\n"
            "300,-50,280,180,15,0,30,L,Stroke1\n"
            "300,0,280,180,15,0,30,L,Stroke2\n"
            "300,50,280,180,15,0,30,L,Stroke3\n"
            "300,100,280,180,15,0,30,L,Stroke4\n"
            "300,0,450,180,0,0,50,L,Retract\n"
        )
