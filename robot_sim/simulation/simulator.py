"""
PyBullet simulation manager for FANUC LR Mate 200iD/14L.

Handles robot loading, joint control, and physics stepping.
"""
import os
import time
import threading
from typing import Optional, List, Tuple

import numpy as np

try:
    import pybullet as pb
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False

from ..robot.kinematics import Kinematics
from ..robot.dh_params import DHParams


URDF_PATH = os.path.join(os.path.dirname(__file__), "..", "robot", "urdf", "lrmate200id14l.urdf")


class Simulator:
    """
    PyBullet simulation manager.

    Loads the FANUC LR Mate 200iD/14L URDF, provides joint control,
    and manages the simulation loop.

    Usage:
        sim = Simulator(gui=True)
        sim.start()
        sim.set_joint_angles([0, -0.5, 0.5, 0, -1.57, 0])
        sim.stop()
    """

    JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

    def __init__(self, gui: bool = False, gravity: float = -9.81):
        """
        Args:
            gui     : If True, open PyBullet GUI window.
            gravity : Gravity in m/s^2 (default -9.81).
        """
        if not PYBULLET_AVAILABLE:
            raise ImportError(
                "pybullet is not installed. Run: pip install pybullet"
            )

        self.gui = gui
        self.gravity = gravity
        self.client_id: int = -1
        self.robot_id: int = -1
        self.joint_indices: List[int] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._kinematics = Kinematics()
        self._current_angles = np.zeros(6)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Initialize PyBullet and load the robot."""
        mode = pb.GUI if self.gui else pb.DIRECT
        self.client_id = pb.connect(mode)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        pb.setGravity(0, 0, self.gravity, physicsClientId=self.client_id)

        # Load ground plane
        pb.loadURDF("plane.urdf", physicsClientId=self.client_id)

        # Load robot URDF
        urdf_path = os.path.abspath(URDF_PATH)
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        self.robot_id = pb.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0],
            baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            physicsClientId=self.client_id,
        )

        # Map joint names to indices
        self.joint_indices = self._find_joint_indices()

        # Set initial pose to ready position
        ready = self._kinematics.dh.ready_position()
        self.set_joint_angles(ready, immediate=True)

        self._running = True

        if self.gui:
            self._configure_gui()

    def stop(self):
        """Disconnect from PyBullet."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.client_id >= 0:
            try:
                pb.disconnect(physicsClientId=self.client_id)
            except Exception:
                pass
        self.client_id = -1
        self.robot_id = -1

    def is_running(self) -> bool:
        return self._running and self.client_id >= 0

    # ------------------------------------------------------------------
    # Joint control
    # ------------------------------------------------------------------

    def set_joint_angles(
        self,
        angles: np.ndarray,
        immediate: bool = False,
        max_velocity: float = 1.0,
    ):
        """
        Set robot joint angles.

        Args:
            angles      : 6 joint angles in radians.
            immediate   : If True, teleport joints (no motion control).
            max_velocity: Maximum joint velocity for position control (rad/s).
        """
        angles = np.asarray(angles, dtype=float)
        if len(angles) != 6:
            raise ValueError(f"Expected 6 angles, got {len(angles)}")

        self._current_angles = angles.copy()

        if not self.is_running():
            return

        if immediate:
            for idx, angle in zip(self.joint_indices, angles):
                pb.resetJointState(
                    self.robot_id, idx, angle,
                    physicsClientId=self.client_id
                )
        else:
            pb.setJointMotorControlArray(
                self.robot_id,
                self.joint_indices,
                pb.POSITION_CONTROL,
                targetPositions=angles.tolist(),
                positionGains=[0.1] * 6,
                velocityGains=[1.0] * 6,
                physicsClientId=self.client_id,
            )

    def get_joint_angles(self) -> np.ndarray:
        """Read current joint angles from simulation."""
        if not self.is_running():
            return self._current_angles.copy()

        states = pb.getJointStates(
            self.robot_id, self.joint_indices,
            physicsClientId=self.client_id
        )
        return np.array([s[0] for s in states])

    def get_end_effector_pose(self) -> np.ndarray:
        """Return current end-effector pose as 4x4 transform (mm)."""
        q = self.get_joint_angles()
        return self._kinematics.forward(q)

    # ------------------------------------------------------------------
    # Motion execution
    # ------------------------------------------------------------------

    def move_to_joint_angles(
        self,
        target_angles: np.ndarray,
        steps: int = 100,
        dt: float = 0.02,
    ) -> bool:
        """
        Smoothly move robot to target joint angles.

        Args:
            target_angles : 6 target joint angles (radians).
            steps         : Number of interpolation steps.
            dt            : Time step between steps (seconds).
        Returns:
            True if motion completed successfully.
        """
        if not self.is_running():
            return False

        start_angles = self.get_joint_angles()
        target_angles = np.asarray(target_angles)

        for i in range(steps + 1):
            alpha = i / steps
            interp = start_angles + alpha * (target_angles - start_angles)
            self.set_joint_angles(interp, immediate=True)
            pb.stepSimulation(physicsClientId=self.client_id)
            time.sleep(dt)

        return True

    def move_to_cartesian_pose(
        self,
        T_target: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        steps: int = 100,
        dt: float = 0.02,
    ) -> bool:
        """
        Move robot to target Cartesian pose using IK.

        Args:
            T_target : 4x4 target transform (mm).
            q_init   : Initial guess for IK (radians).
            steps    : Interpolation steps.
            dt       : Time per step (seconds).
        Returns:
            True if IK succeeded and motion completed.
        """
        q_init = q_init if q_init is not None else self.get_joint_angles()
        q_target, success = self._kinematics.inverse(T_target, q_init)
        if not success:
            return False
        return self.move_to_joint_angles(q_target, steps=steps, dt=dt)

    def execute_route(
        self,
        waypoints: list,
        steps_per_segment: int = 80,
        dt: float = 0.01,
        progress_callback=None,
    ) -> bool:
        """
        Execute a sequence of waypoints.

        Args:
            waypoints         : List of Waypoint objects.
            steps_per_segment : Interpolation steps per waypoint.
            dt                : Time per step.
            progress_callback : Optional callable(idx, total) for progress.
        Returns:
            True if all waypoints were reached.
        """
        from ..path.route import Waypoint

        for i, wp in enumerate(waypoints):
            T = wp.to_transform()
            q_init = self.get_joint_angles()
            q_target, ok = self._kinematics.inverse(T, q_init)
            if not ok:
                print(f"[Simulator] IK failed for waypoint {i}: {wp}")
                continue

            self.move_to_joint_angles(q_target, steps=steps_per_segment, dt=dt)

            if progress_callback:
                progress_callback(i + 1, len(waypoints))

        return True

    # ------------------------------------------------------------------
    # Step simulation
    # ------------------------------------------------------------------

    def step(self):
        """Advance simulation by one step."""
        if self.is_running():
            pb.stepSimulation(physicsClientId=self.client_id)

    def reset(self):
        """Reset robot to home position."""
        home = self._kinematics.dh.home_position()
        self.set_joint_angles(home, immediate=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_joint_indices(self) -> List[int]:
        """Find PyBullet joint indices matching JOINT_NAMES."""
        n = pb.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        name_to_idx = {}
        for i in range(n):
            info = pb.getJointInfo(self.robot_id, i, physicsClientId=self.client_id)
            joint_name = info[1].decode("utf-8")
            name_to_idx[joint_name] = i

        indices = []
        for name in self.JOINT_NAMES:
            if name not in name_to_idx:
                raise RuntimeError(f"Joint '{name}' not found in URDF. Available: {list(name_to_idx.keys())}")
            indices.append(name_to_idx[name])
        return indices

    def _configure_gui(self):
        """Configure the PyBullet GUI camera and settings."""
        pb.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.5],
            physicsClientId=self.client_id,
        )
        pb.configureDebugVisualizer(
            pb.COV_ENABLE_GUI, 1,
            physicsClientId=self.client_id
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
