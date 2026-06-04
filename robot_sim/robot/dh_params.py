"""
DH Parameters for FANUC LR Mate 200iD/14L
14L = Long-arm variant, reach ~911mm

Modified DH convention (Craig notation):
  a_i-1   : link length (mm)
  alpha_i-1: link twist (deg)
  d_i     : link offset (mm)
  theta_i : joint angle (variable) + offset (deg)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DHJoint:
    """Single joint DH parameters."""
    a: float          # link length (mm)
    alpha: float      # link twist (deg)
    d: float          # link offset (mm)
    theta_offset: float  # joint angle offset (deg)
    joint_min: float  # joint limit min (deg)
    joint_max: float  # joint limit max (deg)
    name: str = ""

    @property
    def alpha_rad(self) -> float:
        return np.deg2rad(self.alpha)

    @property
    def theta_offset_rad(self) -> float:
        return np.deg2rad(self.theta_offset)

    def joint_min_rad(self) -> float:
        return np.deg2rad(self.joint_min)

    def joint_max_rad(self) -> float:
        return np.deg2rad(self.joint_max)


class DHParams:
    """
    DH Parameters for FANUC LR Mate 200iD/14L.

    Uses Modified DH (MDH) convention as is standard for FANUC robots.
    All linear dimensions in mm, angular in degrees.
    """

    # Robot physical constants
    REACH_MM = 911.0          # Maximum reach
    PAYLOAD_KG = 14.0         # Payload capacity
    REPEATABILITY_MM = 0.02   # Positioning repeatability

    def __init__(self):
        self.joints: List[DHJoint] = self._build_joints()

    def _build_joints(self) -> List[DHJoint]:
        """
        Build DH parameter table for all 6 joints.

        Modified DH parameters — Z-up base frame convention.
        At home (q=all zeros) the arm extends horizontally in +X.
        (a_mm, alpha_deg, d_mm, theta_offset_deg)

          J1: a=0,   alpha=0,    d=330, offset=0   base rotation (Z vertical)
          J2: a=75,  alpha=-90,  d=0,   offset=0   shoulder
          J3: a=450, alpha=0,    d=0,   offset=0   upper arm (14L long variant)
          J4: a=75,  alpha=-90,  d=450, offset=0   forearm  (14L long variant)
          J5: a=0,   alpha=90,   d=0,   offset=0   wrist pitch
          J6: a=0,   alpha=-90,  d=80,  offset=0   wrist roll + flange
        """
        return [
            DHJoint(a=0,   alpha=0,   d=330, theta_offset=0, joint_min=-170, joint_max=170,  name="J1"),
            DHJoint(a=75,  alpha=-90, d=0,   theta_offset=0, joint_min=-85,  joint_max=145,  name="J2"),
            DHJoint(a=450, alpha=0,   d=0,   theta_offset=0, joint_min=-175, joint_max=255,  name="J3"),
            DHJoint(a=75,  alpha=-90, d=450, theta_offset=0, joint_min=-190, joint_max=190,  name="J4"),
            DHJoint(a=0,   alpha=90,  d=0,   theta_offset=0, joint_min=-135, joint_max=135,  name="J5"),
            DHJoint(a=0,   alpha=-90, d=80,  theta_offset=0, joint_min=-360, joint_max=360,  name="J6"),
        ]

    @property
    def num_joints(self) -> int:
        return len(self.joints)

    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (lower_limits_rad, upper_limits_rad)."""
        lower = np.array([j.joint_min_rad() for j in self.joints])
        upper = np.array([j.joint_max_rad() for j in self.joints])
        return lower, upper

    def get_joint_limits_deg(self) -> Tuple[List[float], List[float]]:
        """Returns (lower_limits_deg, upper_limits_deg)."""
        lower = [j.joint_min for j in self.joints]
        upper = [j.joint_max for j in self.joints]
        return lower, upper

    def home_position(self) -> np.ndarray:
        """Return home/zero joint angles (radians)."""
        return np.zeros(6)

    def ready_position(self) -> np.ndarray:
        """Return a 'ready' pose — arm raised diagonally, suitable for sharpening work."""
        # J2=-45° (肩を上方45°), J3=+30° (肘を前方へ), J5=-60° (手首やや下向き)
        return np.deg2rad([0, -45, 30, 0, -60, 0])

    def __repr__(self) -> str:
        lines = ["DHParams for FANUC LR Mate 200iD/14L", "-" * 50,
                 f"{'Joint':6} {'a(mm)':8} {'alpha':8} {'d(mm)':8} {'offset':8} {'min':8} {'max':8}"]
        for j in self.joints:
            lines.append(
                f"{j.name:6} {j.a:8.1f} {j.alpha:8.1f} {j.d:8.1f} "
                f"{j.theta_offset:8.1f} {j.joint_min:8.1f} {j.joint_max:8.1f}"
            )
        return "\n".join(lines)
