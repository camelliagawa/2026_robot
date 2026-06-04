"""User frame (UFrame) definition for FANUC."""
from dataclasses import dataclass
import numpy as np


@dataclass
class UserFrame:
    """
    Represents a FANUC UFrame (User Frame) definition.

    The user frame defines a coordinate system in world space that waypoints
    are expressed relative to. All positions in mm, angles in degrees.
    """
    number: int = 0
    name: str = "WORLD"
    x: float = 0.0    # mm from world origin
    y: float = 0.0
    z: float = 0.0
    rx: float = 0.0   # degrees (ZYX Euler)
    ry: float = 0.0
    rz: float = 0.0
    comment: str = ""

    def to_transform(self) -> np.ndarray:
        """Return 4x4 homogeneous transform from world to user frame origin."""
        from .kinematics import Kinematics
        return Kinematics.pose_to_transform(self.x, self.y, self.z,
                                            self.rx, self.ry, self.rz)

    @classmethod
    def world(cls) -> "UserFrame":
        """World frame (identity — same as robot base frame)."""
        return cls(number=0, name="WORLD",
                   comment="World coordinate frame")

    @classmethod
    def default_stone(cls) -> "UserFrame":
        """Whetstone user frame: 400mm forward (+X), 200mm up (+Z)."""
        return cls(number=1, name="STONE", x=400.0, z=200.0,
                   comment="Whetstone surface")

    def __repr__(self) -> str:
        return (
            f"UserFrame(#{self.number} '{self.name}': "
            f"x={self.x:.1f}, y={self.y:.1f}, z={self.z:.1f} mm, "
            f"rx={self.rx:.1f}, ry={self.ry:.1f}, rz={self.rz:.1f} deg)"
        )
