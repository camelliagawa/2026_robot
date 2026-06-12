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
from typing import Dict, List, Optional, Tuple

import numpy as np

from .route import Route, Waypoint, MotionType
from ..robot.kinematics import Kinematics


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

    def export(self, route: Route, file_path: str,
               utool: Optional[int] = None,
               uframe: Optional[int] = None,
               speed_override: int = 100,
               uframe_pos: Optional[Tuple[float, ...]] = None,
               utool_pos: Optional[Tuple[float, ...]] = None,
               cart_pos: bool = False,
               ideal_attr: bool = False,
               preserve_name_case: bool = False,
               mn_comments: Optional[List[str]] = None,
               joint_solutions: Optional[Dict[int, np.ndarray]] = None):
        """
        Export route to .ls file.

        Args:
            route          : Route object to export.
            file_path      : Output .ls file path.
            utool          : Override UTool number (uses route.utool if None).
            uframe         : Override UFrame number (uses route.uframe if None).
            speed_override : Speed override percentage (1–100).
            uframe_pos     : (x,y,z,w,p,r) to embed UFRAME PR definition.
            utool_pos      : (x,y,z,w,p,r) to embed UTOOL PR definition.
            cart_pos       : Emit Cartesian /POS entries (X/Y/Z/W/P/R in the
                             user frame) for LINEAR waypoints, like RoboDK.
            ideal_attr     : Mimic the RoboDK-generated /ATTR block
                             (PROG_SIZE=0, MEMORY_SIZE=0, no OVERRIDE line).
            preserve_name_case : Keep program name case as given (e.g. "HaL").
            mn_comments    : Optional "!" comment lines placed before motions.
            joint_solutions: {waypoint_index: q_rad} precomputed IK solutions
                             to use instead of solving.
        """
        content = self.generate(route, utool=utool, uframe=uframe,
                                speed_override=speed_override,
                                uframe_pos=uframe_pos, utool_pos=utool_pos,
                                cart_pos=cart_pos, ideal_attr=ideal_attr,
                                preserve_name_case=preserve_name_case,
                                mn_comments=mn_comments,
                                joint_solutions=joint_solutions)
        with open(file_path, "w", encoding="ascii", errors="replace") as f:
            f.write(content)

    def generate(self, route: Route,
                 utool: Optional[int] = None,
                 uframe: Optional[int] = None,
                 speed_override: int = 100,
                 uframe_pos: Optional[Tuple[float, ...]] = None,
                 utool_pos: Optional[Tuple[float, ...]] = None,
                 cart_pos: bool = False,
                 ideal_attr: bool = False,
                 preserve_name_case: bool = False,
                 mn_comments: Optional[List[str]] = None,
                 joint_solutions: Optional[Dict[int, np.ndarray]] = None) -> str:
        """Generate TP program as string.

        Args:
            uframe_pos: (x,y,z,w,p,r) for UFRAME PR register setup. If provided,
                        adds PR[n]=... and UFRAME[n]=PR[n] lines (like HaL.LS).
            utool_pos:  (x,y,z,w,p,r) for UTOOL PR register setup.
            cart_pos:   Emit Cartesian /POS entries for LINEAR waypoints,
                        expressed in the user frame:
                        P = inv(T_uframe) @ T_flange @ T_utool.
            ideal_attr: Mimic the RoboDK-generated /ATTR block.
            preserve_name_case: Keep the program name case as given.
            mn_comments: Optional "!" comment lines placed before motions.
        """
        if not route.waypoints:
            raise ValueError("Route has no waypoints to export.")

        # Override frame numbers if provided
        if utool is not None:
            route.utool = utool
        if uframe is not None:
            route.uframe = uframe

        # Solve IK (only where joint values are needed in cart_pos mode)
        joint_angles = self._solve_ik_all(route, cart_pos=cart_pos,
                                          joint_solutions=joint_solutions)

        # Build sections
        prog_name = self._sanitize_name(route.name,
                                        preserve_case=preserve_name_case)
        now = datetime.datetime.now()
        date_str = now.strftime("%y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        mn_lines = self._build_mn_section(route, joint_angles,
                                          speed_override=speed_override,
                                          uframe_pos=uframe_pos,
                                          utool_pos=utool_pos,
                                          emit_override=not ideal_attr,
                                          mn_comments=mn_comments)
        pos_section = self._build_pos_section(route, joint_angles,
                                              cart_pos=cart_pos,
                                              uframe_pos=uframe_pos,
                                              utool_pos=utool_pos,
                                              pos_comments=not ideal_attr)
        line_count = len(mn_lines)

        lines = []
        lines.append(f"/PROG  {prog_name}")
        lines.append("/ATTR")
        lines.append(f"OWNER\t\t= MNEDITOR;")
        lines.append(f'COMMENT\t\t= "{route.comment[:24]}";')
        if ideal_attr:
            lines.append("PROG_SIZE\t= 0;")
        else:
            lines.append(f"PROG_SIZE\t= {max(512, line_count * 20)};")
        lines.append(f"CREATE\t\t= DATE {date_str}  TIME {time_str};")
        lines.append(f"MODIFIED\t= DATE {date_str}  TIME {time_str};")
        lines.append(f"FILE_NAME\t= {prog_name};")
        lines.append(f"VERSION\t\t= 0;")
        lines.append(f"LINE_COUNT\t= {line_count};")
        if ideal_attr:
            lines.append("MEMORY_SIZE\t= 0;")
        else:
            lines.append(f"MEMORY_SIZE\t= {max(1024, line_count * 50)};")
        lines.append(f"PROTECT\t\t= READ_WRITE;")
        if not ideal_attr:
            lines.append(f"; SPEED_OVERRIDE = {speed_override}%;")
            lines.append(f"; UTOOL = {route.utool}, UFRAME = {route.uframe};")
        lines.append("TCD:  STACK_SIZE\t= 0,")
        lines.append("      TASK_PRIORITY\t= 50,")
        lines.append("      TIME_SLICE\t= 0,")
        lines.append("      BUSY_LAMP_OFF\t= 0,")
        lines.append("      ABORT_REQUEST\t= 0,")
        lines.append("      PAUSE_REQUEST\t= 0;")
        if ideal_attr:
            lines.append("DEFAULT_GROUP\t= 1,*,*,*,*;")
        else:
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

    def _solve_ik_all(self, route: Route,
                      cart_pos: bool = False,
                      joint_solutions: Optional[Dict[int, np.ndarray]] = None,
                      ) -> List[Optional[np.ndarray]]:
        """
        Solve IK for each waypoint.

        In cart_pos mode IK is only solved for JOINT-type waypoints (the
        Cartesian /POS entries do not need joint values). CALL entries are
        skipped (None). Precomputed solutions in joint_solutions
        ({waypoint_index: q_rad}) are used as-is.

        Returns list of joint angle arrays (radians), or None where not needed.
        """
        results = []
        q_prev = self.kin.dh.ready_position()  # start from ready pose

        for i, wp in enumerate(route.waypoints):
            if wp.call is not None:
                results.append(None)
                continue
            if joint_solutions is not None and i in joint_solutions:
                q_given = np.asarray(joint_solutions[i], dtype=float)
                results.append(q_given)
                q_prev = q_given
                continue
            if cart_pos and wp.motion_type != MotionType.JOINT:
                results.append(None)
                continue
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
        self, route: Route, joint_angles: List[Optional[np.ndarray]],
        speed_override: int = 100,
        uframe_pos: Optional[Tuple[float, ...]] = None,
        utool_pos: Optional[Tuple[float, ...]] = None,
        emit_override: bool = True,
        mn_comments: Optional[List[str]] = None,
    ) -> List[str]:
        """Build the /MN motion instruction lines."""
        lines = []
        line_num = 1
        uf = route.uframe
        ut = route.utool

        # UFrame PR register setup (like HaL.LS PR[9,1..6] = ...; UFRAME[9]=PR[9])
        if uframe_pos and len(uframe_pos) >= 6:
            labels = ["1", "2", "3", "4", "5", "6"]
            for i, val in enumerate(uframe_pos[:6]):
                lines.append(f"{line_num:4d}:  PR[{uf},{labels[i]}]={val:.3f} ;")
                line_num += 1
            lines.append(f"{line_num:4d}:  UFRAME[{uf}]=PR[{uf}] ;")
            line_num += 1
        lines.append(f"{line_num:4d}:  UFRAME_NUM={uf} ;")
        line_num += 1

        # UTool PR register setup
        if utool_pos and len(utool_pos) >= 6:
            labels = ["1", "2", "3", "4", "5", "6"]
            for i, val in enumerate(utool_pos[:6]):
                lines.append(f"{line_num:4d}:  PR[{ut},{labels[i]}]={val:.3f} ;")
                line_num += 1
            lines.append(f"{line_num:4d}:  UTOOL[{ut}]=PR[{ut}] ;")
            line_num += 1
        lines.append(f"{line_num:4d}:  UTOOL_NUM={ut} ;")
        line_num += 1

        if emit_override:
            lines.append(f"{line_num:4d}:  OVERRIDE={speed_override}% ;")
            line_num += 1

        # Optional "!" comment lines (like RoboDK header comments)
        for c in (mn_comments or []):
            lines.append(f"{line_num:4d}:  ! {c[:24]} ;")
            line_num += 1

        p_idx = 0  # P[1]-based indexing (CALL entries consume no P index)
        for wp in route.waypoints:
            if wp.call is not None:
                lines.append(f"{line_num:4d}:  CALL {wp.call} ;")
                line_num += 1
                continue
            p_idx += 1
            motion_str = self._motion_instruction(wp, p_idx,
                                                   speed_override=speed_override)
            lines.append(f"{line_num:4d}:{motion_str}  ;")
            line_num += 1

        return lines

    @staticmethod
    def _term_str(wp: Waypoint) -> str:
        """Termination type string: CNTn if wp.cnt is set, else FINE."""
        return f"CNT{int(wp.cnt)}" if wp.cnt is not None else "FINE"

    def _motion_instruction(self, wp: Waypoint, p_idx: int,
                             speed_override: int = 100) -> str:
        """Format a single motion instruction."""
        scale = max(1, min(100, speed_override)) / 100.0
        term = self._term_str(wp)
        if wp.motion_type == MotionType.JOINT:
            if wp.joint_speed_pct is not None:
                pct = int(min(100, max(1, wp.joint_speed_pct)))
            else:
                pct = int(min(100, max(1, wp.speed / 5.0 * scale)))
            return f"J P[{p_idx}] {pct}% {term}"
        elif wp.motion_type == MotionType.CIRCULAR:
            return f"C P[{p_idx}]"  # CIRCULAR needs two points; simplified
        else:
            speed_str = f"{int(wp.speed * scale)}mm/sec"
            return f"L P[{p_idx}] {speed_str} {term}"

    # ------------------------------------------------------------------
    # /POS section
    # ------------------------------------------------------------------

    def _build_pos_section(
        self, route: Route, joint_angles: List[Optional[np.ndarray]],
        cart_pos: bool = False,
        uframe_pos: Optional[Tuple[float, ...]] = None,
        utool_pos: Optional[Tuple[float, ...]] = None,
        pos_comments: bool = True,
    ) -> List[str]:
        """Build the /POS position definitions.

        Joint format for JOINT waypoints (or always, when cart_pos=False);
        Cartesian X/Y/Z/W/P/R (+CONFIG) format for the rest when
        cart_pos=True, expressed in the user frame:
            P = inv(T_uframe) @ T_flange @ T_utool
        """
        lines = []
        uframe = route.uframe
        utool = route.utool

        # User frame / tool transforms for Cartesian output
        if uframe_pos and len(uframe_pos) >= 6:
            T_uf_inv = np.linalg.inv(
                Kinematics.pose_to_transform(*uframe_pos[:6]))
        else:
            T_uf_inv = np.eye(4)
        if utool_pos and len(utool_pos) >= 6:
            T_ut = Kinematics.pose_to_transform(*utool_pos[:6])
        else:
            T_ut = np.eye(4)

        p_idx = 0
        for wp, q in zip(route.waypoints, joint_angles):
            if wp.call is not None:
                continue  # CALL entries have no position
            p_idx += 1
            comment = f"  ; {wp.label}" if (wp.label and pos_comments) else ""

            if cart_pos and wp.motion_type != MotionType.JOINT:
                # Cartesian position in user frame coordinates
                T_pose = T_uf_inv @ wp.to_transform() @ T_ut
                x, y, z, w, p, r = Kinematics.transform_to_pose(T_pose)
                lines.append(f"P[{p_idx}]{{")
                lines.append(f"   GP1:")
                lines.append(f"    UF : {uframe}, UT : {utool},"
                             f"\t\tCONFIG : 'F U T, 0, 0, 0',{comment}")
                lines.append(
                    f"\tX = {x:9.3f}  mm,\tY = {y:9.3f}  mm,\tZ = {z:9.3f}  mm,"
                )
                lines.append(
                    f"\tW = {w:9.3f} deg,\tP = {p:9.3f} deg,\tR = {r:9.3f} deg"
                )
                lines.append("};")
                continue

            if q is None:
                q = np.zeros(6)
            q_deg = np.rad2deg(q)

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
    def _sanitize_name(name: str, max_len: int = 24,
                       preserve_case: bool = False) -> str:
        """Sanitize program name for FANUC TP (alphanumeric + underscore, max 24).

        preserve_case=True keeps the given case (e.g. "HaL" like RoboDK output);
        otherwise the name is uppercased (legacy behaviour).
        """
        if not preserve_case:
            name = name.upper()
        sanitized = "".join(
            c if (c.isalnum() or c == "_") else "_"
            for c in name
        )
        return sanitized[:max_len] if sanitized else "KNIFE_ROUTE"
