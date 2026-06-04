"""
FANUC TP (.ls) program exporter for knife sharpening routes.

Generates a FANUC TP language file (.ls) that can be loaded onto
a FANUC robot controller via FTP or USB.

Reference: FANUC Robotics SYSTEM R-30iB/R-30iB Plus Controller
           Karel and TP Language Reference Manual.

Output format example:
  /PROG  KNIFE_ROUTE
  /ATTR
  OWNER          = MNEDITOR;
  COMMENT        = "Knife sharpening route";
  PROG_SIZE      = 512;
  CREATE         = DATE 25-01-01  TIME 00:00:00;
  MODIFIED       = DATE 25-01-01  TIME 00:00:00;
  FILE_NAME      = KNIFE_ROUTE;
  VERSION        = 0;
  LINE_COUNT     = 6;
  MEMORY_SIZE    = 1024;
  PROTECT        = READ_WRITE;
  TCD:  STACK_SIZE    = 0,
        TASK_PRIORITY = 50,
        TIME_SLICE    = 0,
        BUSY_LAMP_OFF = 0,
        ABORT_REQUEST = 0,
        PAUSE_REQUEST = 0;
  DEFAULT_GROUP  = 1, *,  *,  *,  *;
  CONTROL_CODE   = 00000000 00000000;
  /MN
     1:  UFRAME_NUM=0 ;
     2:  UTOOL_NUM=1 ;
     3:J P[1] 30% FINE    ;
     4:L P[2] 50mm/sec FINE    ;
  /POS
  P[1]{
     GP1:
    UF : 0, UT : 1,
    J1 = 0.000 deg,    J2 = 0.000 deg,    J3 = 0.000 deg,
    J4 = 0.000 deg,    J5 = -90.000 deg,  J6 = 0.000 deg
  };
  /END
"""
from __future__ import annotations

import datetime
from typing import List, Optional, Tuple

import numpy as np

from .route import Route, Waypoint, MotionType
from ..robot.kinematics import Kinematics
from ..robot.dh_params import DHParams


class TPExporter:
    """
    Export a Route to FANUC TP (.ls) program format.

    The exporter:
      1. Runs IK for each waypoint to compute joint angles.
      2. Generates the /MN section with motion instructions.
      3. Generates the /POS section with joint-space positions.
    """

    def __init__(self, kinematics: Optional[Kinematics] = None):
        self.kin = kinematics or Kinematics()

    # ------------------------------------------------------------------
    # Main export
    # ------------------------------------------------------------------

    def export(self, route: Route, file_path: str):
        """
        Export route to .ls file.

        Args:
            route     : Route object to export.
            file_path : Output .ls file path.
        """
        content = self.generate(route)
        with open(file_path, "w", encoding="ascii", errors="replace") as f:
            f.write(content)

    def generate(self, route: Route) -> str:
        """Generate TP program as string."""
        if not route.waypoints:
            raise ValueError("Route has no waypoints to export.")

        # Solve IK for all waypoints
        joint_angles = self._solve_ik_all(route)

        # Build sections
        prog_name = self._sanitize_name(route.name)
        now = datetime.datetime.now()
        date_str = now.strftime("%y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        line_count = len(route.waypoints) + 2  # +2 for UFRAME/UTOOL lines

        mn_lines = self._build_mn_section(route, joint_angles)
        pos_section = self._build_pos_section(route, joint_angles)

        lines = []
        lines.append(f"/PROG  {prog_name}")
        lines.append("/ATTR")
        lines.append(f"OWNER\t\t= MNEDITOR;")
        lines.append(f'COMMENT\t\t= "{route.comment[:24]}";')
        lines.append(f"PROG_SIZE\t= {max(512, line_count * 20)};")
        lines.append(f"CREATE\t\t= DATE {date_str}  TIME {time_str};")
        lines.append(f"MODIFIED\t= DATE {date_str}  TIME {time_str};")
        lines.append(f"FILE_NAME\t= {prog_name};")
        lines.append(f"VERSION\t\t= 0;")
        lines.append(f"LINE_COUNT\t= {line_count};")
        lines.append(f"MEMORY_SIZE\t= {max(1024, line_count * 50)};")
        lines.append(f"PROTECT\t\t= READ_WRITE;")
        lines.append("TCD:  STACK_SIZE    = 0,")
        lines.append("      TASK_PRIORITY = 50,")
        lines.append("      TIME_SLICE    = 0,")
        lines.append("      BUSY_LAMP_OFF = 0,")
        lines.append("      ABORT_REQUEST = 0,")
        lines.append("      PAUSE_REQUEST = 0;")
        lines.append("DEFAULT_GROUP\t= 1, *,  *,  *,  *;")
        lines.append("CONTROL_CODE\t= 00000000 00000000;")
        lines.append("/MN")
        lines.extend(mn_lines)
        lines.append("/POS")
        lines.extend(pos_section)
        lines.append("/END")
        lines.append("")  # trailing newline

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # IK solving
    # ------------------------------------------------------------------

    def _solve_ik_all(self, route: Route) -> List[Optional[np.ndarray]]:
        """
        Solve IK for each waypoint.

        Returns list of joint angle arrays (radians), or None for failures.
        """
        results = []
        q_prev = self.kin.dh.ready_position()  # start from ready pose

        for i, wp in enumerate(route.waypoints):
            T = wp.to_transform()
            q, ok = self.kin.inverse(T, q_init=q_prev)
            if ok:
                results.append(q)
                q_prev = q
            else:
                # Use previous angles as fallback to keep sequence consistent
                print(f"[TPExporter] Warning: IK failed for waypoint {i} ({wp.label or 'unnamed'}). "
                      f"Using previous joint angles.")
                results.append(q_prev.copy())

        return results

    # ------------------------------------------------------------------
    # /MN section
    # ------------------------------------------------------------------

    def _build_mn_section(
        self, route: Route, joint_angles: List[Optional[np.ndarray]]
    ) -> List[str]:
        """Build the /MN motion instruction lines."""
        lines = []
        line_num = 1

        # Standard FANUC program header lines
        lines.append(f"{line_num:4d}:  UFRAME_NUM={route.uframe} ;")
        line_num += 1
        lines.append(f"{line_num:4d}:  UTOOL_NUM={route.utool} ;")
        line_num += 1

        for i, wp in enumerate(route.waypoints):
            p_idx = i + 1  # P[1]-based indexing
            motion_str = self._motion_instruction(wp, p_idx)
            lines.append(f"{line_num:4d}:{motion_str}    ;")
            line_num += 1

        return lines

    def _motion_instruction(self, wp: Waypoint, p_idx: int) -> str:
        """Format a single motion instruction."""
        if wp.motion_type == MotionType.JOINT:
            speed_str = f"{int(min(100, max(1, wp.speed / 5.0)))}%"
            return f"J P[{p_idx}] {speed_str} FINE"
        elif wp.motion_type == MotionType.LINEAR:
            speed_str = f"{int(wp.speed)}mm/sec"
            return f"L P[{p_idx}] {speed_str} FINE"
        elif wp.motion_type == MotionType.CIRCULAR:
            speed_str = f"{int(wp.speed)}mm/sec"
            return f"C P[{p_idx}]"  # CIRCULAR needs two points; simplified
        else:
            speed_str = f"{int(wp.speed)}mm/sec"
            return f"L P[{p_idx}] {speed_str} FINE"

    # ------------------------------------------------------------------
    # /POS section
    # ------------------------------------------------------------------

    def _build_pos_section(
        self, route: Route, joint_angles: List[Optional[np.ndarray]]
    ) -> List[str]:
        """Build the /POS joint position definitions."""
        lines = []
        uframe = route.uframe
        utool = route.utool

        for i, (wp, q) in enumerate(zip(route.waypoints, joint_angles)):
            p_idx = i + 1
            if q is None:
                q = np.zeros(6)

            q_deg = np.rad2deg(q)
            comment = f"  ; {wp.label}" if wp.label else ""

            lines.append(f"P[{p_idx}]{{")
            lines.append(f"   GP1:")
            lines.append(f"\tUF : {uframe}, UT : {utool},{comment}")
            lines.append(
                f"\tJ1 = {q_deg[0]:8.3f} deg,\t"
                f"J2 = {q_deg[1]:8.3f} deg,\t"
                f"J3 = {q_deg[2]:8.3f} deg,"
            )
            lines.append(
                f"\tJ4 = {q_deg[3]:8.3f} deg,\t"
                f"J5 = {q_deg[4]:8.3f} deg,\t"
                f"J6 = {q_deg[5]:8.3f} deg"
            )
            lines.append("};")

        return lines

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_name(name: str, max_len: int = 24) -> str:
        """Sanitize program name for FANUC TP (alphanumeric + underscore, max 24)."""
        sanitized = "".join(
            c if (c.isalnum() or c == "_") else "_"
            for c in name.upper()
        )
        return sanitized[:max_len] if sanitized else "KNIFE_ROUTE"

    def preview(self, route: Route) -> str:
        """Return generated TP content as string (same as generate)."""
        return self.generate(route)
