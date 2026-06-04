"""Tool frame (UTool) definition for FANUC."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ToolFrame:
    """
    Represents a FANUC UTool (Tool Frame) definition.

    The tool frame defines the transformation from the robot flange (J6 flange)
    to the Tool Center Point (TCP). All positions in mm, angles in degrees.
    """
    number: int = 1
    name: str = "TOOL1"
    x: float = 0.0    # mm from flange origin
    y: float = 0.0
    z: float = 200.0   # knife blade center: 200mm from flange along Z
    rx: float = 0.0    # degrees (ZYX Euler)
    ry: float = 0.0
    rz: float = 0.0
    comment: str = ""

    def to_transform(self) -> np.ndarray:
        """Return 4x4 homogeneous transform from flange to TCP."""
        from .kinematics import Kinematics
        return Kinematics.pose_to_transform(self.x, self.y, self.z,
                                            self.rx, self.ry, self.rz)

    @classmethod
    def default_knife(cls) -> "ToolFrame":
        """Default knife tool: blade center extends 200mm along Z from flange."""
        return cls(number=1, name="KNIFE", z=200.0,
                   comment="Kitchen knife blade")

    @classmethod
    def flange(cls) -> "ToolFrame":
        """Identity tool at flange (no offset)."""
        return cls(number=0, name="FLANGE", z=0.0,
                   comment="Flange (no tool offset)")

    def __repr__(self) -> str:
        return (
            f"ToolFrame(#{self.number} '{self.name}': "
            f"x={self.x:.1f}, y={self.y:.1f}, z={self.z:.1f} mm, "
            f"rx={self.rx:.1f}, ry={self.ry:.1f}, rz={self.rz:.1f} deg)"
        )
