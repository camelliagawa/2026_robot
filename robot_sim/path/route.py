"""
Route data model: Waypoints and Route for knife sharpening.

A Route is an ordered list of Waypoints, each specifying:
  - 3D position (x, y, z) in mm
  - Orientation as ZYX Euler angles (rx, ry, rz) in degrees
  - Motion speed (mm/s)
  - Motion type: JOINT, LINEAR (L), or CIRCULAR (C)
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

import numpy as np


class MotionType(str, Enum):
    """FANUC motion types."""
    JOINT = "J"       # Joint interpolation
    LINEAR = "L"      # Linear Cartesian interpolation
    CIRCULAR = "C"    # Circular arc interpolation


@dataclass
class Waypoint:
    """
    A single pose in the robot's route.

    Attributes:
        x, y, z     : Position in mm (base frame)
        rx, ry, rz  : ZYX Euler orientation in degrees
        speed       : Speed in mm/s (LINEAR) or % (JOINT)
        motion_type : MotionType enum
        label       : Optional name/comment
        id          : Unique identifier
        cnt         : FANUC termination type — None = FINE, int n = CNTn
        joint_speed_pct : Explicit % speed for JOINT motions (overrides speed)
        call        : If set, this entry is a "CALL <prog>" instruction
                      instead of a motion (no /POS entry is generated)
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 400.0
    rx: float = 180.0
    ry: float = 0.0
    rz: float = 0.0
    speed: float = 50.0
    motion_type: MotionType = MotionType.LINEAR
    label: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    cnt: Optional[int] = None
    joint_speed_pct: Optional[int] = None
    call: Optional[str] = None

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def to_transform(self) -> np.ndarray:
        """Return 4x4 homogeneous transform (mm)."""
        from ..robot.kinematics import Kinematics
        return Kinematics.pose_to_transform(
            self.x, self.y, self.z,
            self.rx, self.ry, self.rz
        )

    def position(self) -> np.ndarray:
        """Return position as (3,) array in mm."""
        return np.array([self.x, self.y, self.z])

    def __str__(self) -> str:
        return (
            f"WP[{self.id}] ({self.x:.1f}, {self.y:.1f}, {self.z:.1f}) mm  "
            f"RPY=({self.rx:.1f}, {self.ry:.1f}, {self.rz:.1f}) deg  "
            f"{self.motion_type.value} {self.speed:.0f}mm/s"
        )


class Route:
    """
    Ordered collection of Waypoints representing a knife sharpening path.

    Attributes:
        name        : Program name (used in TP export)
        comment     : Optional comment
        waypoints   : Ordered list of Waypoint
        uframe      : User frame number (FANUC)
        utool       : User tool number (FANUC)
    """

    def __init__(
        self,
        name: str = "KNIFE_ROUTE",
        comment: str = "Knife sharpening route",
        uframe: int = 0,
        utool: int = 1,
    ):
        self.name = name
        self.comment = comment
        self.uframe = uframe
        self.utool = utool
        self.waypoints: List[Waypoint] = []

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_waypoint(self, wp: Waypoint, index: Optional[int] = None):
        """Append or insert waypoint."""
        if index is None:
            self.waypoints.append(wp)
        else:
            self.waypoints.insert(index, wp)

    def remove_waypoint(self, index: int):
        """Remove waypoint at index."""
        if 0 <= index < len(self.waypoints):
            self.waypoints.pop(index)

    def move_waypoint(self, from_idx: int, to_idx: int):
        """Reorder waypoints."""
        if not (0 <= from_idx < len(self.waypoints)):
            return
        wp = self.waypoints.pop(from_idx)
        to_idx = max(0, min(to_idx, len(self.waypoints)))
        self.waypoints.insert(to_idx, wp)

    def clear(self):
        """Remove all waypoints."""
        self.waypoints.clear()

    def __len__(self) -> int:
        return len(self.waypoints)

    def __getitem__(self, idx: int) -> Waypoint:
        return self.waypoints[idx]

    def __iter__(self):
        return iter(self.waypoints)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def total_length_mm(self) -> float:
        """Approximate total path length (straight-line segments) in mm."""
        if len(self.waypoints) < 2:
            return 0.0
        positions = np.array([wp.position() for wp in self.waypoints])
        deltas = np.diff(positions, axis=0)
        return float(np.sum(np.linalg.norm(deltas, axis=1)))

    def estimated_time_sec(self) -> float:
        """Estimate total motion time assuming each segment runs at its speed."""
        if len(self.waypoints) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self.waypoints)):
            d = np.linalg.norm(
                self.waypoints[i].position() - self.waypoints[i - 1].position()
            )
            speed = max(self.waypoints[i].speed, 1.0)
            total += d / speed
        return total

    def positions_array(self) -> np.ndarray:
        """Return (N, 3) array of waypoint positions."""
        if not self.waypoints:
            return np.zeros((0, 3))
        return np.array([wp.position() for wp in self.waypoints])

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def default_sharpening_route(cls) -> "Route":
        """
        Create a sample knife sharpening route.

        Minimal sample: Home → Approach → Retract (no stroke waypoints).
        """
        route = cls(name="KNIFE_SHARPEN", comment="Knife sharpening demo route")

        # Coordinate convention: X=forward, Y=lateral(left+), Z=height(up)
        # Whetstone is placed ~400mm forward, ~250mm height from base

        # Home approach (above stone)
        route.add_waypoint(Waypoint(
            x=400, y=0, z=350, rx=180, ry=0, rz=0,
            speed=30, motion_type=MotionType.JOINT, label="Home"
        ))

        # Approach stone surface
        route.add_waypoint(Waypoint(
            x=400, y=-80, z=250, rx=180, ry=15, rz=0,
            speed=50, motion_type=MotionType.LINEAR, label="Approach"
        ))

        # Retract above stone
        route.add_waypoint(Waypoint(
            x=400, y=0, z=350, rx=180, ry=0, rz=0,
            speed=50, motion_type=MotionType.LINEAR, label="Retract"
        ))

        return route

    def __repr__(self) -> str:
        return (
            f"Route(name='{self.name}', waypoints={len(self.waypoints)}, "
            f"length={self.total_length_mm():.1f}mm)"
        )
