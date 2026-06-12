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
from dataclasses import dataclass
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
    joint_max_speed: float = 0.0  # deg/s — maximum joint speed
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
    PAYLOAD_KG = 14.0           # Payload capacity (kg)
    REACH_MM = 911.0            # Maximum reach to wrist center (mm)
    REPEATABILITY_MM = 0.02     # Positioning repeatability (mm)
    WEIGHT_KG = 27.0            # Robot weight (kg)
    CONTROLLER = "R-30iB Plus"  # Controller model
    IP_RATING = "IP67"          # Protection rating

    def __init__(self):
        self.joints: List[DHJoint] = self._build_joints()

    def _build_joints(self) -> List[DHJoint]:
        """
        Build DH parameter table for all 6 joints.

        Modified DH parameters — Z-up base frame convention.
        Dimensions derived from ROS-Industrial URDF (fanuc_lrmate200id_support)
        and scaled to 200iD/14L (911mm reach, same kinematics as 7L variant).

        URDF joint origins (metres → mm):
          joint_1: xyz="0 0 0.330"   → d1 = 330
          joint_2: xyz="0.050 0 0"   → a1 = 50
          joint_3: xyz="0 0 0.440"   → a2 = 440  (7L/14L long-arm)
          joint_4: xyz="0 0 0.035"   → a3 = 35
          joint_5: xyz="0.420 0 0"   → d4 = 420  (7L/14L long-arm)
          joint_6: xyz="0.080 0 0"   → d6 = 80

        Verified reach:
          max reach to wrist center (J6 axis) = 911mm ✓  (FANUC spec)
          max reach to flange (tool0)         = 991mm    (wrist + d6=80mm)
          FANUC reach spec excludes the flange offset d6.

        Standard 200iD:  a2=330, d4=335 → reach 717mm
        7L / 14L:        a2=440, d4=420 → reach 911mm  (same kinematics)
        """
        return [
            DHJoint(a=0,   alpha=0,   d=330, theta_offset=0, joint_min=-170, joint_max=170,  joint_max_speed=210, name="J1"),
            DHJoint(a=50,  alpha=-90, d=0,   theta_offset=0, joint_min=-85,  joint_max=145,  joint_max_speed=210, name="J2"),
            DHJoint(a=440, alpha=0,   d=0,   theta_offset=0, joint_min=-175, joint_max=255,  joint_max_speed=275, name="J3"),
            DHJoint(a=35,  alpha=-90, d=420, theta_offset=0, joint_min=-190, joint_max=190,  joint_max_speed=400, name="J4"),
            DHJoint(a=0,   alpha=90,  d=0,   theta_offset=0, joint_min=-135, joint_max=135,  joint_max_speed=400, name="J5"),
            DHJoint(a=0,   alpha=-90, d=80,  theta_offset=0, joint_min=-360, joint_max=360,  joint_max_speed=600, name="J6"),
        ]

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
        # J2=-70° (肩を上方), J3=+20° (肘を前方へ), J4=+5°, J5=-60° (手首やや下向き)
        return np.deg2rad([0, -70, 20, 5, -60, 0])

    def get_joint_max_speeds(self) -> List[float]:
        """Returns list of max speeds (deg/s) for each joint."""
        return [j.joint_max_speed for j in self.joints]

    def __repr__(self) -> str:
        lines = [
            "DHParams for FANUC LR Mate 200iD/14L", "-" * 70,
            f"  Payload: {self.PAYLOAD_KG} kg  |  Reach: {self.REACH_MM} mm  |  "
            f"Repeatability: ±{self.REPEATABILITY_MM} mm",
            f"  Weight: {self.WEIGHT_KG} kg  |  Controller: {self.CONTROLLER}  |  "
            f"Protection: {self.IP_RATING}",
            "-" * 70,
            f"{'Joint':6} {'a(mm)':8} {'alpha':8} {'d(mm)':8} {'offset':8} "
            f"{'min':8} {'max':8} {'spd°/s':8}",
        ]
        for j in self.joints:
            lines.append(
                f"{j.name:6} {j.a:8.1f} {j.alpha:8.1f} {j.d:8.1f} "
                f"{j.theta_offset:8.1f} {j.joint_min:8.1f} {j.joint_max:8.1f} "
                f"{j.joint_max_speed:8.1f}"
            )
        return "\n".join(lines)
