"""Parse FANUC TP (.ls) files and convert to Route objects.

Supports:
- Single /PROG sections
- Multiple /PROG sections in one file (e.g. kenma.LS with HaL + HaR inline)
- Joint-space positions (J1-J6) converted via FK to Cartesian Waypoints
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np

from .route import Route, Waypoint, MotionType


def ls_to_route(path: str, kinematics=None) -> List[Route]:
    """Parse a FANUC .ls file and return one Route per /PROG section.

    Joint positions in /POS are converted to Cartesian via FK so every
    Waypoint carries a valid (x,y,z,rx,ry,rz) pose.

    Args:
        path:        Absolute path to the .ls file.
        kinematics:  Kinematics instance (created if None).

    Returns:
        List of Route objects — typically 1, or 2+ for inline multi-prog files.
    """
    if kinematics is None:
        from ..robot.kinematics import Kinematics
        kinematics = Kinematics()

    with open(path, "r", encoding="ascii", errors="replace") as f:
        content = f.read()

    routes: List[Route] = []
    # Split into /PROG … /END blocks (keep delimiter with lookahead)
    sections = re.split(r"(?=^/PROG\b)", content, flags=re.MULTILINE)

    for section in sections:
        if not section.strip().startswith("/PROG"):
            continue
        route = _parse_prog_section(section, kinematics)
        if route and route.waypoints:
            routes.append(route)

    return routes


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_prog_section(text: str, kin) -> Optional[Route]:
    """Parse one /PROG … /END block into a Route."""
    m = re.search(r"^/PROG\s+(\S+)", text, re.MULTILINE)
    if not m:
        return None
    prog_name = m.group(1)

    cm = re.search(r'^COMMENT\s*=\s*"([^"]*)"', text, re.MULTILINE)
    comment = cm.group(1) if cm else ""

    uf_m = re.search(r"UFRAME_NUM\s*=\s*(\d+)", text)
    ut_m = re.search(r"UTOOL_NUM\s*=\s*(\d+)", text)
    uframe = int(uf_m.group(1)) if uf_m else 0
    utool  = int(ut_m.group(1)) if ut_m else 1

    pos_m = re.search(r"/POS(.*?)(?=/END|$)", text, re.DOTALL)
    if not pos_m:
        return None

    joint_positions = _parse_pos_section(pos_m.group(1))
    if not joint_positions:
        return None

    mn_m = re.search(r"/MN(.*?)(?=/POS|/END|$)", text, re.DOTALL)
    motions = _parse_mn_section(mn_m.group(1)) if mn_m else {}

    route = Route()
    route.name    = prog_name
    route.comment = comment or f"Imported from {prog_name}"
    route.uframe  = uframe
    route.utool   = utool

    for p_idx in sorted(joint_positions.keys()):
        q_deg = joint_positions[p_idx]
        q = np.deg2rad(q_deg)
        T = kin.forward(q)
        x, y, z, rx, ry, rz = kin.transform_to_pose(T)

        motion_type, speed = motions.get(p_idx, (MotionType.JOINT, 30.0))
        wp = Waypoint(
            x=x, y=y, z=z, rx=rx, ry=ry, rz=rz,
            speed=speed, motion_type=motion_type,
            label=f"P[{p_idx}]",
        )
        route.waypoints.append(wp)

    return route


def _parse_pos_section(text: str) -> dict:
    """Return {p_idx: [j1..j6 degrees]} from /POS block."""
    result: dict = {}
    for b in re.finditer(r"P\[(\d+)\]\{(.*?)\};", text, re.DOTALL):
        p_idx  = int(b.group(1))
        joints = re.findall(r"J\d+\s*=\s*([-\d.]+)\s*deg", b.group(2))
        if len(joints) == 6:
            result[p_idx] = [float(j) for j in joints]
    return result


def _parse_mn_section(text: str) -> dict:
    """Return {p_idx: (MotionType, speed)} from /MN block."""
    result: dict = {}
    for line in text.splitlines():
        line = line.strip()
        # J P[n] nn%  or  J P[n] nn% FINE
        m = re.search(r"\bJ\s+P\[(\d+)\]\s+(\d+)%", line)
        if m:
            result[int(m.group(1))] = (MotionType.JOINT, float(m.group(2)) * 5.0)
            continue
        # L P[n] nn.nmm/sec
        m = re.search(r"\bL\s+P\[(\d+)\]\s+(\d+(?:\.\d+)?)mm/sec", line)
        if m:
            result[int(m.group(1))] = (MotionType.LINEAR, float(m.group(2)))
    return result
