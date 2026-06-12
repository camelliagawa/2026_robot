"""
曲線追従の姿勢計算で使う共有ジオメトリヘルパー。

これらの関数は kenma_export（kenma形式LS生成）の姿勢計算規約で
共有される（法線・接線からの右手系基底構築、軸まわり回転など）。
"""
from __future__ import annotations

import numpy as np


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
