"""
Automatic knife sharpening route generator.

Given whetstone dimensions and knife parameters, generates a Route
that sweeps the knife blade across the stone surface.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .route import Route, Waypoint, MotionType


@dataclass
class SharpeningParams:
    """Parameters for automatic sharpening route generation."""
    # Whetstone position (center of top surface) in robot base frame, mm
    stone_x: float = 400.0
    stone_y: float = 0.0
    stone_z: float = 250.0

    # Whetstone dimensions (mm)
    stone_length: float = 200.0   # along Y axis (knife sliding direction)
    stone_width: float = 70.0     # along X axis

    # Knife geometry
    blade_angle_deg: float = 15.0   # approach angle of blade to stone (tilt)
    blade_length_mm: float = 180.0  # usable blade length to sharpen

    # Motion parameters
    approach_height_mm: float = 80.0   # height above stone for approach/retract
    stroke_speed_mms: float = 30.0     # sharpening stroke speed (mm/s)
    approach_speed_mms: float = 80.0   # approach/retract speed (mm/s)
    num_strokes: int = 5               # number of forward+back stroke cycles
    stroke_overlap_mm: float = 10.0    # overlap between adjacent strokes along X

    # Route metadata
    route_name: str = "KNIFE_SHARPEN"
    route_comment: str = "Auto-generated sharpening route"
    utool: int = 1
    uframe: int = 0


def generate_sharpening_route(p: SharpeningParams) -> Route:
    """
    Generate a knife sharpening route from SharpeningParams.

    The knife strokes run along the Y-axis (stone length direction).
    Multiple passes are distributed along the X-axis (stone width direction)
    to sharpen the full blade length.
    """
    route = Route(name=p.route_name, comment=p.route_comment,
                  utool=p.utool, uframe=p.uframe)

    rx = 180.0  # TCP pointing down
    ry = p.blade_angle_deg  # blade tilt

    # X positions for each pass (distributed across stone width)
    n_passes = max(1, int(p.blade_length_mm / (p.stone_width - p.stroke_overlap_mm)))
    if n_passes == 1:
        x_positions = [p.stone_x]
    else:
        x_start = p.stone_x - p.stone_width / 2 + p.stroke_overlap_mm
        x_end = p.stone_x + p.stone_width / 2 - p.stroke_overlap_mm
        x_positions = list(np.linspace(x_start, x_end, n_passes))

    y_start = p.stone_y - p.stone_length / 2
    y_end = p.stone_y + p.stone_length / 2

    # 1. Home approach — above stone center
    route.add_waypoint(Waypoint(
        x=p.stone_x, y=p.stone_y,
        z=p.stone_z + p.approach_height_mm,
        rx=rx, ry=0.0, rz=0.0,
        speed=p.approach_speed_mms,
        motion_type=MotionType.JOINT,
        label="HOME",
    ))

    stroke_count = 0
    for pass_idx, px in enumerate(x_positions):
        # Approach to near start of stroke
        route.add_waypoint(Waypoint(
            x=px, y=y_start,
            z=p.stone_z + p.approach_height_mm,
            rx=rx, ry=ry, rz=0.0,
            speed=p.approach_speed_mms,
            motion_type=MotionType.LINEAR,
            label=f"APPR_{pass_idx+1}",
        ))
        # Lower to stone surface
        route.add_waypoint(Waypoint(
            x=px, y=y_start,
            z=p.stone_z,
            rx=rx, ry=ry, rz=0.0,
            speed=p.approach_speed_mms / 2,
            motion_type=MotionType.LINEAR,
            label=f"TOUCH_{pass_idx+1}",
        ))

        for stroke in range(p.num_strokes):
            stroke_count += 1
            # Forward stroke
            route.add_waypoint(Waypoint(
                x=px, y=y_end,
                z=p.stone_z,
                rx=rx, ry=ry, rz=0.0,
                speed=p.stroke_speed_mms,
                motion_type=MotionType.LINEAR,
                label=f"FWD_{pass_idx+1}_{stroke+1}",
            ))
            # Backward stroke
            route.add_waypoint(Waypoint(
                x=px, y=y_start,
                z=p.stone_z,
                rx=rx, ry=ry, rz=0.0,
                speed=p.stroke_speed_mms,
                motion_type=MotionType.LINEAR,
                label=f"BWD_{pass_idx+1}_{stroke+1}",
            ))

        # Lift off stone
        route.add_waypoint(Waypoint(
            x=px, y=y_start,
            z=p.stone_z + p.approach_height_mm,
            rx=rx, ry=ry, rz=0.0,
            speed=p.approach_speed_mms,
            motion_type=MotionType.LINEAR,
            label=f"LIFT_{pass_idx+1}",
        ))

    # Return to home
    route.add_waypoint(Waypoint(
        x=p.stone_x, y=p.stone_y,
        z=p.stone_z + p.approach_height_mm,
        rx=rx, ry=0.0, rz=0.0,
        speed=p.approach_speed_mms,
        motion_type=MotionType.JOINT,
        label="HOME_RTN",
    ))

    return route
