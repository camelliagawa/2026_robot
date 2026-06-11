"""
曲線を辿るプロジェクト（RoboDK の Curve Follow Project 相当）。

刃先CSV（ブレードローカル座標の点列 + 法線）の各点を順番に
砥石上の研削接触点（参照フレーム原点）へ接触させる
刃付けルートを自動生成する。

座標系:
  - blade_pts / blade_normals : ブレードローカル座標（刃渡り = 局所 +Y）
  - T_blade  : フランジ → ブレードローカルの取付オフセット (4x4)
  - T_contact: ワールド座標での接触参照フレーム (4x4, +Z=砥石面法線)
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .route import Waypoint, MotionType


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _rotation_about_axis(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues の回転公式による軸まわり回転行列。"""
    a = _normalize(axis)
    th = np.radians(angle_deg)
    K = np.array([[0, -a[2], a[1]],
                  [a[2], 0, -a[0]],
                  [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _orthonormal_basis(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """u を第1軸とし、v を Gram-Schmidt 直交化した右手系基底 (3x3, 列ベクトル)。"""
    e1 = _normalize(u)
    v2 = v - np.dot(v, e1) * e1
    if np.linalg.norm(v2) < 1e-9:
        # v が u と平行な場合のフォールバック
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, e1)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        v2 = tmp - np.dot(tmp, e1) * e1
    e2 = _normalize(v2)
    e3 = np.cross(e1, e2)
    return np.column_stack([e1, e2, e3])


def _tangent(pts: np.ndarray, i: int) -> np.ndarray:
    """点列に沿った接線方向（中央差分、端は片側差分）。"""
    n = len(pts)
    j0 = max(0, i - 1)
    j1 = min(n - 1, i + 1)
    return _normalize(pts[j1] - pts[j0])


def generate_curve_follow(
    blade_pts: np.ndarray,
    blade_normals: np.ndarray,
    T_blade: np.ndarray,
    T_contact: np.ndarray,
    kin=None,
    *,
    edge_angle_deg: float = 15.0,
    step: int = 1,
    speed: float = 30.0,
    approach_mm: float = 30.0,
) -> Tuple[List[Waypoint], int]:
    """
    刃先の各点を砥石接触点へ順に接触させるウェイポイント列を生成する。

    Returns:
        (waypoints, n_unreachable)
    """
    blade_pts = np.asarray(blade_pts, dtype=float)
    blade_normals = np.asarray(blade_normals, dtype=float)
    step = max(1, int(step))

    C = T_contact[:3, 3]            # 接触点（参照フレーム原点）
    s = _normalize(T_contact[:3, 2])  # 砥石面法線（+Z, 上向き）
    x_stone = T_contact[:3, 0]      # 砥石面内の接線方向（+X）

    T_blade_inv = np.linalg.inv(T_blade)

    waypoints: List[Waypoint] = []
    n_unreachable = 0
    q_seed = None
    indices = list(range(0, len(blade_pts), step))

    contact_wps: List[Tuple[int, np.ndarray]] = []  # (元index, T_ee)

    for i in indices:
        p = blade_pts[i]
        n_i = _normalize(blade_normals[i])
        t_i = _tangent(blade_pts, i)

        # ソース基底 {n_i, t_i⊥, n×t} → ターゲット基底 {-s, x_stone⊥, cross}
        B_src = _orthonormal_basis(n_i, t_i)
        B_tgt = _orthonormal_basis(-s, x_stone)
        R_bw = B_tgt @ B_src.T

        # 刃付け角度: ワールド接線軸まわりに傾ける
        if abs(edge_angle_deg) > 1e-9:
            t_world = R_bw @ t_i
            R_bw = _rotation_about_axis(t_world, edge_angle_deg) @ R_bw

        # 並進: R_bw @ p + trans = C
        T_bw = np.eye(4)
        T_bw[:3, :3] = R_bw
        T_bw[:3, 3] = C - R_bw @ p

        T_ee = T_bw @ T_blade_inv
        contact_wps.append((i, T_ee))

    def _make_wp(T_ee, label, motion_type) -> Waypoint:
        x, y, z, rx, ry, rz = _transform_to_pose(kin, T_ee)
        return Waypoint(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz,
                        speed=speed, motion_type=motion_type, label=label)

    def _reachable(T_ee) -> bool:
        nonlocal q_seed
        if kin is None:
            return True
        q, ok = kin.inverse(T_ee, q_init=q_seed)
        if ok and q is not None:
            q_seed = q
        return ok

    out: List[Waypoint] = []
    first_T = None
    last_T = None
    for i, T_ee in contact_wps:
        if not _reachable(T_ee):
            n_unreachable += 1
            continue
        if first_T is None:
            first_T = T_ee
        last_T = T_ee
        out.append(_make_wp(T_ee, f"CF[{i}]", MotionType.LINEAR))

    if first_T is not None and approach_mm > 0:
        T_app = first_T.copy()
        T_app[:3, 3] = T_app[:3, 3] + approach_mm * s
        out.insert(0, _make_wp(T_app, "CF_APPROACH", MotionType.JOINT))
    if last_T is not None and approach_mm > 0:
        T_ret = last_T.copy()
        T_ret[:3, 3] = T_ret[:3, 3] + approach_mm * s
        out.append(_make_wp(T_ret, "CF_RETRACT", MotionType.LINEAR))

    waypoints = out
    return waypoints, n_unreachable


def _transform_to_pose(kin, T: np.ndarray):
    """Kinematics.transform_to_pose のラッパ（kin=None でも動作）。"""
    if kin is not None:
        return kin.transform_to_pose(T)
    from ..robot.kinematics import Kinematics
    return Kinematics.transform_to_pose(T)
