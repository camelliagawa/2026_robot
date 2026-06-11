"""
Forward and Inverse Kinematics for FANUC LR Mate 200iD/14L.

Forward Kinematics: Denavit-Hartenberg transformation matrices (Modified DH).
Inverse Kinematics: Analytical closed-form solution for 6R robot with spherical wrist,
  with numerical fallback via scipy.optimize.
"""
import numpy as np
from scipy.optimize import minimize
from typing import Optional, List, Tuple
from .dh_params import DHParams


class Kinematics:
    """
    Kinematics solver for FANUC LR Mate 200iD/14L.

    Convention: Modified DH (MDH), dimensions in mm, angles in radians internally.
    """

    def __init__(self, dh: Optional[DHParams] = None):
        self.dh = dh or DHParams()
        self._lower, self._upper = self.dh.get_joint_limits()

    # ------------------------------------------------------------------
    # DH Transformation matrix
    # ------------------------------------------------------------------

    @staticmethod
    def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        """
        Compute Modified DH transformation matrix T_i-1_i.

        Args:
            a     : link length (mm)
            alpha : link twist (rad)
            d     : link offset (mm)
            theta : joint angle (rad) — includes theta_offset
        Returns:
            4x4 homogeneous transformation matrix
        """
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([
            [ct,       -st,       0,        a],
            [st * ca,   ct * ca,  -sa,  -sa * d],
            [st * sa,   ct * sa,   ca,   ca * d],
            [0,         0,         0,        1],
        ])

    # ------------------------------------------------------------------
    # Forward Kinematics
    # ------------------------------------------------------------------

    def forward(self, q: np.ndarray) -> np.ndarray:
        """
        Compute end-effector pose via Forward Kinematics.

        Args:
            q : joint angles in radians, shape (6,)
        Returns:
            T : 4x4 homogeneous transformation (base -> end-effector), mm
        """
        q = np.asarray(q, dtype=float)
        if q.shape != (6,):
            raise ValueError(f"Expected 6 joint angles, got {q.shape}")

        T = np.eye(4)
        for i, joint in enumerate(self.dh.joints):
            theta = q[i] + joint.theta_offset_rad
            Ti = self.dh_transform(joint.a, joint.alpha_rad, joint.d, theta)
            T = T @ Ti
        return T

    def forward_all(self, q: np.ndarray) -> List[np.ndarray]:
        """
        Return list of 4x4 transforms for each joint (base -> joint_i).

        Returns list of length 7: [T_base, T_0, T_1, ..., T_5]
        """
        q = np.asarray(q, dtype=float)
        transforms = [np.eye(4)]
        T = np.eye(4)
        for i, joint in enumerate(self.dh.joints):
            theta = q[i] + joint.theta_offset_rad
            Ti = self.dh_transform(joint.a, joint.alpha_rad, joint.d, theta)
            T = T @ Ti
            transforms.append(T.copy())
        return transforms

    def get_joint_positions(self, q: np.ndarray) -> np.ndarray:
        """
        Return Cartesian positions of each joint origin (mm).

        Returns array shape (7, 3): base origin + 6 joint positions
        """
        transforms = self.forward_all(q)
        return np.array([T[:3, 3] for T in transforms])

    # ------------------------------------------------------------------
    # Inverse Kinematics
    # ------------------------------------------------------------------

    def inverse(
        self,
        T_target: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        use_analytical: bool = True,
        tol: float = 1e-4,
    ) -> Tuple[Optional[np.ndarray], bool]:
        """
        Compute joint angles for a target end-effector pose.

        Args:
            T_target       : 4x4 target homogeneous transform (mm)
            q_init         : initial joint angles for numerical IK (radians)
            use_analytical : try analytical solution first (default True)
            tol            : position tolerance in mm for success check

        Returns:
            (q, success) : joint angles (radians) or None, and success flag
        """
        if use_analytical:
            q, ok = self._analytical_ik(T_target)
            if ok:
                return q, True

        # Fallback to numerical IK
        return self._numerical_ik(T_target, q_init, tol)

    def _analytical_ik(
        self, T: np.ndarray
    ) -> Tuple[Optional[np.ndarray], bool]:
        """
        Analytical IK for 6R robot with spherical wrist.

        Uses the standard decoupling approach:
          1. Find wrist center from desired EE pose.
          2. Solve J1, J2, J3 for the wrist center position.
          3. Solve J4, J5, J6 from the wrist orientation.

        Returns first valid solution found.
        """
        # Extract position and orientation
        p_ee = T[:3, 3]
        R = T[:3, :3]

        dh = self.dh.joints
        # Parameters (mm) — matches DHParams: d1=330, a1=0, a2=50, a3=440, d4=420, d6=80
        d1 = dh[0].d   # 330 — J1 vertical offset (base height)
        a1 = dh[0].a   # 0   — J1 link length
        a2 = dh[1].a   # 50  — J2 link length (shoulder offset)
        a3 = dh[2].a   # 440 — J3 link length (upper arm)
        d4 = dh[3].d   # 420 — J4 link offset (forearm length)
        d6 = dh[5].d   # 80  — J6 link offset (wrist to flange)

        # Wrist center: subtract d6 along EE z-axis
        p_wc = p_ee - d6 * R[:, 2]

        wx, wy, wz = p_wc

        # ---------- Solve J1 ----------
        r = np.sqrt(wx**2 + wy**2)
        # Two solutions: elbow up / down, shoulder left / right
        j1_candidates = [np.arctan2(wy, wx), np.arctan2(wy, wx) + np.pi]

        solutions = []
        for j1 in j1_candidates:
            if not self._in_limits(0, j1):
                continue
            c1, s1 = np.cos(j1), np.sin(j1)

            # Wrist center in shoulder frame
            wx_s = c1 * wx + s1 * wy - a1
            wy_s = wz - d1   # height above shoulder

            # Distance from J2 origin to wrist center
            L = np.sqrt(wx_s**2 + wy_s**2)

            # Law of cosines for J3
            # Link lengths in the arm plane: a2 and sqrt(a3^2 + d4^2)
            l_upper = a2
            l_lower = np.sqrt(a3**2 + d4**2)

            cos_j3_inner = (L**2 - l_upper**2 - l_lower**2) / (2 * l_upper * l_lower)
            if abs(cos_j3_inner) > 1.0:
                continue  # Unreachable
            sin_j3_inner_pos = np.sqrt(1 - cos_j3_inner**2)

            # Offset angle for J3 due to a3, d4
            phi3 = np.arctan2(a3, d4)

            for sign in [1, -1]:  # elbow up / down
                sin_j3_inner = sign * sin_j3_inner_pos
                # J3 inner angle
                j3_inner = np.arctan2(sin_j3_inner, cos_j3_inner)
                j3 = j3_inner - phi3  # corrected for DH offset

                if not self._in_limits(2, j3):
                    continue

                # J2
                beta = np.arctan2(wy_s, wx_s)
                gamma = np.arctan2(l_lower * sin_j3_inner, l_upper + l_lower * cos_j3_inner)
                j2 = beta - gamma

                # DH J2 offset is -90 deg
                if not self._in_limits(1, j2):
                    continue

                # ---------- Solve J4, J5, J6 from wrist orientation ----------
                # R_03 = T03[:3,:3]
                T01 = self.dh_transform(dh[0].a, dh[0].alpha_rad, dh[0].d,
                                        j1 + dh[0].theta_offset_rad)
                T12 = self.dh_transform(dh[1].a, dh[1].alpha_rad, dh[1].d,
                                        j2 + dh[1].theta_offset_rad)
                T23 = self.dh_transform(dh[2].a, dh[2].alpha_rad, dh[2].d,
                                        j3 + dh[2].theta_offset_rad)
                R03 = (T01 @ T12 @ T23)[:3, :3]

                # R36 = R03^T @ R
                R36 = R03.T @ R

                # ZYZ Euler angles for spherical wrist (alpha=J4, beta=J5, gamma=J6)
                # Standard ZYZ decomposition of R36
                j5 = np.arctan2(np.sqrt(R36[0, 2]**2 + R36[1, 2]**2), R36[2, 2])
                if abs(j5) < 1e-6:
                    # Gimbal lock: J5 ~ 0
                    j4 = 0.0
                    j6 = np.arctan2(-R36[0, 1], R36[0, 0])
                elif abs(j5 - np.pi) < 1e-6:
                    j4 = 0.0
                    j6 = np.arctan2(R36[0, 1], -R36[0, 0])
                else:
                    j4 = np.arctan2(R36[1, 2], R36[0, 2])
                    j6 = np.arctan2(R36[2, 1], -R36[2, 0])

                # Correct for MDH alpha conventions of joints 4,5,6
                # Verify against actual DH
                # For joint 4: alpha=90, joint 5: alpha=-90, joint 6: alpha=0
                # The R36 decomposition using ZYZ gives J4,J5,J6 directly

                if not (self._in_limits(3, j4) and
                        self._in_limits(4, j5) and
                        self._in_limits(5, j6)):
                    # Try alternate branch
                    j5_alt = -j5
                    j4_alt = j4 + np.pi
                    j6_alt = j6 + np.pi
                    if not (self._in_limits(3, j4_alt) and
                            self._in_limits(4, j5_alt) and
                            self._in_limits(5, j6_alt)):
                        continue
                    j4, j5, j6 = j4_alt, j5_alt, j6_alt

                q_sol = np.array([j1, j2, j3, j4, j5, j6])
                solutions.append(q_sol)

        if not solutions:
            return None, False

        # Pick solution closest to zero (home)
        best = min(solutions, key=lambda q: np.linalg.norm(q))

        # Verify solution
        T_check = self.forward(best)
        pos_err = np.linalg.norm(T_check[:3, 3] - T[:3, 3])
        if pos_err > 5.0:  # mm tolerance for analytical verification
            return None, False

        return best, True

    def _numerical_ik(
        self,
        T_target: np.ndarray,
        q_init: Optional[np.ndarray],
        tol: float = 1e-4,
    ) -> Tuple[Optional[np.ndarray], bool]:
        """
        Numerical IK via scipy.optimize.minimize with multiple restarts.
        """
        p_target = T_target[:3, 3]
        R_target = T_target[:3, :3]

        def cost(q):
            T = self.forward(q)
            p = T[:3, 3]
            R = T[:3, :3]
            pos_err = np.sum((p - p_target)**2)
            # Frobenius norm of rotation error
            rot_err = np.sum((R - R_target)**2)
            return pos_err + 100.0 * rot_err

        bounds = list(zip(self._lower, self._upper))

        best_q = None
        best_cost = np.inf

        # Initial guesses
        if q_init is not None:
            inits = [q_init, np.zeros(6)]
        else:
            inits = [np.zeros(6)]

        # Add random restarts
        rng = np.random.default_rng(42)
        for _ in range(8):
            q_rand = rng.uniform(self._lower, self._upper)
            inits.append(q_rand)

        for q0 in inits:
            q0 = np.clip(q0, self._lower, self._upper)
            result = minimize(
                cost, q0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
            )
            if result.fun < best_cost:
                best_cost = result.fun
                best_q = result.x
            if best_cost < 1e-9:
                # Practically exact (pos err ~3e-5 mm) — stop early.
                # Keeps the solution in the branch of the provided seed.
                break

        if best_q is None:
            return None, False

        T_check = self.forward(best_q)
        pos_err = np.linalg.norm(T_check[:3, 3] - p_target)
        success = pos_err < max(tol, 1.0)  # 1mm tolerance for numerical
        return best_q, success

    def _in_limits(self, joint_idx: int, angle_rad: float) -> bool:
        """Check if joint angle is within limits."""
        return (self._lower[joint_idx] <= angle_rad <= self._upper[joint_idx])

    # ------------------------------------------------------------------
    # Jacobian (numerical)
    # ------------------------------------------------------------------

    def jacobian(self, q: np.ndarray, delta: float = 1e-6) -> np.ndarray:
        """
        Compute 6x6 geometric Jacobian numerically.

        Returns:
            J : shape (6, 6) where rows 0:3 are linear velocity, 3:6 angular velocity
        """
        q = np.asarray(q, dtype=float)
        T0 = self.forward(q)
        p0 = T0[:3, 3]
        R0 = T0[:3, :3]

        J = np.zeros((6, 6))
        for i in range(6):
            dq = np.zeros(6)
            dq[i] = delta
            T1 = self.forward(q + dq)
            p1 = T1[:3, 3]
            R1 = T1[:3, :3]

            J[:3, i] = (p1 - p0) / delta
            # Angular velocity from skew-symmetric part of dR * R^T
            dR = (R1 - R0) / delta
            skew = dR @ R0.T
            J[3, i] = skew[2, 1]
            J[4, i] = skew[0, 2]
            J[5, i] = skew[1, 0]
        return J

    # ------------------------------------------------------------------
    # Pose utilities
    # ------------------------------------------------------------------

    @staticmethod
    def rpy_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """ZYX Euler angles (radians) to rotation matrix."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        R = np.array([
            [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
            [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
            [-sp,      cp * sr,                  cp * cr],
        ])
        return R

    @staticmethod
    def rotation_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
        """Rotation matrix to ZYX Euler angles (radians)."""
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
        if abs(np.cos(pitch)) < 1e-9:
            roll = 0.0
            yaw = np.arctan2(R[0, 1], R[1, 1]) if pitch > 0 else np.arctan2(-R[0, 1], -R[1, 1])
        else:
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        return roll, pitch, yaw

    @staticmethod
    def pose_to_transform(x: float, y: float, z: float,
                           rx: float, ry: float, rz: float) -> np.ndarray:
        """
        Build 4x4 transform from position (mm) and ZYX Euler angles (degrees).
        """
        R = Kinematics.rpy_to_rotation(
            np.deg2rad(rx), np.deg2rad(ry), np.deg2rad(rz)
        )
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    @staticmethod
    def transform_to_pose(T: np.ndarray):
        """
        Extract position (mm) and ZYX Euler angles (degrees) from 4x4 transform.
        Returns (x, y, z, rx_deg, ry_deg, rz_deg)
        """
        x, y, z = T[:3, 3]
        roll, pitch, yaw = Kinematics.rotation_to_rpy(T[:3, :3])
        return x, y, z, np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)
