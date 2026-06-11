"""Parse FANUC TP (.ls) files and convert to Route objects.

Supports:
- Single /PROG sections and multiple /PROG sections in one file
- Joint-space positions (J1-J6) converted via FK to Cartesian Waypoints
- Cartesian positions (X,Y,Z,W,P,R) in user frame, transformed to world coords
- PR register parsing in /MN to reconstruct UFrame/UTool
"""
from __future__ import annotations
import re
from typing import List, Optional
import numpy as np
from .route import Route, Waypoint, MotionType


def _parse_pr_registers(text: str) -> dict:
    """Parse PR[n,axis]=value lines from /MN section.
    Returns {reg_num: {1: x, 2: y, 3: z, 4: rx, 5: ry, 6: rz}} in mm/deg.
    """
    registers: dict = {}
    for m in re.finditer(r"PR\[(\d+),(\d+)\]\s*=\s*([-\d.]+)\s*;", text):
        reg = int(m.group(1))
        axis = int(m.group(2))
        val = float(m.group(3))
        registers.setdefault(reg, {})[axis] = val
    return registers


def _pr_to_transform(pr: dict) -> np.ndarray:
    """Convert PR register {1:x,..,6:rz} to 4x4 transform using ZYX Euler."""
    from ..robot.kinematics import Kinematics
    x  = pr.get(1, 0.0); y  = pr.get(2, 0.0); z  = pr.get(3, 0.0)
    rx = pr.get(4, 0.0); ry = pr.get(5, 0.0); rz = pr.get(6, 0.0)
    return Kinematics.pose_to_transform(x, y, z, rx, ry, rz)


def _parse_pos_section(text: str) -> dict:
    """Return {p_idx: {"type": "joint"/"cart", "data": [...]}} from /POS block.
    joint data: [j1..j6 degrees]
    cart  data: [x, y, z, rx(W), ry(P), rz(R)] in mm/deg
    """
    result: dict = {}
    for b in re.finditer(r"P\[(\d+)\]\{(.*?)\};", text, re.DOTALL):
        p_idx = int(b.group(1))
        body  = b.group(2)
        # Joint positions
        joints = re.findall(r"J\d+\s*=\s*([-\d.]+)\s*deg", body)
        if len(joints) == 6:
            result[p_idx] = {"type": "joint", "data": [float(j) for j in joints]}
            continue
        # Cartesian positions
        xm = re.search(r"\bX\s*=\s*([-\d.]+)\s*mm", body)
        ym = re.search(r"\bY\s*=\s*([-\d.]+)\s*mm", body)
        zm = re.search(r"\bZ\s*=\s*([-\d.]+)\s*mm", body)
        wm = re.search(r"\bW\s*=\s*([-\d.]+)\s*deg", body)
        pm = re.search(r"\bP\s*=\s*([-\d.]+)\s*deg", body)
        rm = re.search(r"\bR\s*=\s*([-\d.]+)\s*deg", body)
        if all([xm, ym, zm, wm, pm, rm]):
            result[p_idx] = {"type": "cart", "data": [
                float(xm.group(1)), float(ym.group(1)), float(zm.group(1)),
                float(wm.group(1)), float(pm.group(1)), float(rm.group(1)),
            ]}
    return result


def _parse_mn_section(text: str) -> dict:
    """Return {p_idx: (MotionType, speed)} from /MN block."""
    result: dict = {}
    for line in text.splitlines():
        line = line.strip()
        m = re.search(r"\bJ\s+P\[(\d+)\]\s+(\d+)%", line)
        if m:
            result[int(m.group(1))] = (MotionType.JOINT, float(m.group(2)) * 5.0)
            continue
        m = re.search(r"\bL\s+P\[(\d+)\]\s+(\d+(?:\.\d+)?)mm/sec", line)
        if m:
            result[int(m.group(1))] = (MotionType.LINEAR, float(m.group(2)))
    return result


def _parse_prog_section(text: str, kin) -> Optional[Route]:
    """Parse one /PROG … /END block into a Route."""
    m = re.search(r"^/PROG\s+(\S+)", text, re.MULTILINE)
    if not m:
        return None
    prog_name = m.group(1)

    cm = re.search(r'^COMMENT\s*=\s*"([^"]*)"', text, re.MULTILINE)
    comment = cm.group(1) if cm else ""

    # UFrame / UTool from ATTR header
    uf_m = re.search(r"UFRAME_NUM\s*=\s*(\d+)", text)
    ut_m = re.search(r"UTOOL_NUM\s*=\s*(\d+)",  text)
    uframe_num = int(uf_m.group(1)) if uf_m else 0
    utool_num  = int(ut_m.group(1)) if ut_m else 1

    mn_m   = re.search(r"/MN(.*?)(?=/POS|/END|$)", text, re.DOTALL)
    pos_m  = re.search(r"/POS(.*?)(?=/END|$)",      text, re.DOTALL)
    if not pos_m:
        return None

    mn_text  = mn_m.group(1) if mn_m else ""
    pos_text = pos_m.group(1)

    # Parse PR registers from /MN to reconstruct UFrame / UTool transforms
    pr_regs = _parse_pr_registers(mn_text)
    T_uf = np.eye(4)
    uframe_pr_m = re.search(r"UFRAME\[(\d+)\]\s*=\s*PR\[(\d+)\]", mn_text)
    utool_pr_m  = re.search(r"UTOOL\[(\d+)\]\s*=\s*PR\[(\d+)\]",  mn_text)
    if uframe_pr_m:
        pr_num = int(uframe_pr_m.group(2))
        if pr_num in pr_regs:
            T_uf = _pr_to_transform(pr_regs[pr_num])
    if utool_pr_m:
        pr_num = int(utool_pr_m.group(2))
        if pr_num in pr_regs:
            _pr_to_transform(pr_regs[pr_num])  # parsed but not applied at route level

    motions   = _parse_mn_section(mn_text)
    positions = _parse_pos_section(pos_text)
    if not positions:
        return None

    route         = Route()
    route.name    = prog_name
    route.comment = comment or f"Imported from {prog_name}"
    route.uframe  = uframe_num
    route.utool   = utool_num

    for p_idx in sorted(positions.keys()):
        entry = positions[p_idx]
        motion_type, speed = motions.get(p_idx, (MotionType.JOINT, 30.0))

        if entry["type"] == "joint":
            q_deg = entry["data"]
            q     = np.deg2rad(q_deg)
            T     = kin.forward(q)
        else:
            # Cartesian in UFrame → transform to world
            xc, yc, zc, wc, pc, rc = entry["data"]
            T_local = kin.pose_to_transform(xc, yc, zc, wc, pc, rc)
            T = T_uf @ T_local

        x, y, z, rx, ry, rz = kin.transform_to_pose(T)
        wp = Waypoint(
            x=x, y=y, z=z, rx=rx, ry=ry, rz=rz,
            speed=speed, motion_type=motion_type,
            label=f"P[{p_idx}]",
        )
        route.waypoints.append(wp)

    return route


def ls_to_route(path: str, kinematics=None) -> List[Route]:
    """Parse a FANUC .ls file and return one Route per /PROG section.

    Joint positions converted to Cartesian via FK.
    Cartesian positions (X,Y,Z,W,P,R) converted using UFrame from /MN PR registers.

    Args:
        path:       Absolute path to the .ls file.
        kinematics: Kinematics instance (created if None).
    Returns:
        List of Route objects.
    """
    if kinematics is None:
        from ..robot.kinematics import Kinematics
        kinematics = Kinematics()

    # Security: resolve and validate the path is a real file, not a traversal
    import os
    path = os.path.realpath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"LS file not found: {path}")

    with open(path, "r", encoding="ascii", errors="replace") as f:
        content = f.read()

    routes: List[Route] = []
    sections = re.split(r"(?=^/PROG\b)", content, flags=re.MULTILINE)
    for section in sections:
        if not section.strip().startswith("/PROG"):
            continue
        route = _parse_prog_section(section, kinematics)
        if route and route.waypoints:
            routes.append(route)
    return routes
