"""
kenma形式 3ファイル1組 LSプログラム生成（RoboDK 生成の理想LSセット相当）。

刃先CSV（304行 = 2側面 × 19グループ × 8行）から、砥石接触点上で
「揺動ストローク」を行う per-station 刃付けプログラムを生成する:

  - HaL  : 左側面（nx<0）の 19 グループ × 8 ポイント = 152 位置
  - HaR  : 右側面（nx>0）の同等プログラム
  - kenma: メインプログラム（J ホーム → CALL HaL → J ホーム →
           J HaR側安全姿勢 → CALL HaR → J 安全姿勢 → J ホーム）

CSVグループ構造（連続重複行で汎用検出）:
  各グループ8行のうち 行1==行2、行7==行8（重複）。
  ユニーク接触点 = 行2〜7 の6点（接触開始 + ストローク5点）。
  法線は約3°→20°と傾き、肩から刃先端まで接触が掃引する。

グループごとの動作パターン（reference/HaL.LS と同一）:
  - ホバー   : 接触開始姿勢を砥石面法線方向に +10mm — L 1000mm/sec CNT1
              （先頭グループのみ J P[1] 20% CNT1 の関節動作に置換）
  - 接触開始 : L 1000mm/sec CNT1
  - ストローク5点 : L 50mm/sec CNT1
  - リトラクト: 最終接触姿勢 +10mm — L 1000mm/sec CNT1

姿勢の計算は curve_follow.generate_curve_follow と同じ規約
（刃法線を砥石法線と反平行に整列 + 接線まわりの任意刃付け角）。
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .route import Route, Waypoint, MotionType
from .curve_follow import _normalize, _orthonormal_basis, _rotation_about_axis

# ── 定数（reference/HaL.LS に準拠） ───────────────────────────────
HOVER_MM       = 10.0    # ホバー / リトラクトの離隔距離 [mm]
CONTACT_SPEED  = 1000.0  # ホバー・接触開始・リトラクト速度 [mm/s]
STROKE_SPEED   = 50.0    # ストローク速度 [mm/s]
JOINT_PCT      = 20      # 関節動作速度 [%]
CNT_VAL        = 1       # CNT1
UF_NUM         = 9       # UF9: STONE
UT_NUM         = 9       # UT9
UTOOL9_POS     = (0.0, 0.0, 0.0, -90.0, 0.0, 90.0)  # 理想LSの UTOOL[9]


# ──────────────────────────────────────────────────────────────────
# パス→ツール オフセット式（transl/rotx/roty/rotz の積）の安全パーサ
# ──────────────────────────────────────────────────────────────────

_POSE_FACTOR_RE = re.compile(
    r"^(transl|rotx|roty|rotz)\s*\(\s*([^()]*?)\s*\)$", re.IGNORECASE)


def _rot_elem(axis: int, deg: float) -> np.ndarray:
    """X/Y/Z 軸まわり回転の 4x4 同次変換。"""
    e = np.zeros(3)
    e[axis] = 1.0
    T = np.eye(4)
    T[:3, :3] = _rotation_about_axis(e, deg)
    return T


def parse_pose_expression(expr: str) -> np.ndarray:
    """`transl(x,y,z)*rotx(d)*roty(d)*rotz(d)` 形式の式を 4x4 行列に変換する。

    RoboDK の「パスからツールへのオフセット」と同じ記法。
    eval は使わず正規表現トークナイザで安全に解析する。
    mm / deg、大文字小文字・空白は許容、`*` 区切りの積のみ対応。

    Raises:
        ValueError: 形式不正・未知の関数・引数個数不一致のとき。
    """
    if expr is None:
        raise ValueError("式が空です")
    s = expr.strip()
    if not s:
        raise ValueError("式が空です")
    T = np.eye(4)
    for raw in s.split("*"):
        part = raw.strip()
        m = _POSE_FACTOR_RE.match(part)
        if not m:
            raise ValueError(f"不正な項です: {part!r}  "
                             "(transl(x,y,z) / rotx(d) / roty(d) / rotz(d) のみ対応)")
        func = m.group(1).lower()
        args_str = m.group(2).strip()
        try:
            args = ([float(a) for a in args_str.split(",")]
                    if args_str else [])
        except ValueError:
            raise ValueError(f"数値を解析できません: {part!r}")
        if func == "transl":
            if len(args) != 3:
                raise ValueError(f"transl は引数3個 (x,y,z mm) が必要です: {part!r}")
            Ti = np.eye(4)
            Ti[:3, 3] = args
        else:
            if len(args) != 1:
                raise ValueError(f"{func} は引数1個 (deg) が必要です: {part!r}")
            axis = {"rotx": 0, "roty": 1, "rotz": 2}[func]
            Ti = _rot_elem(axis, args[0])
        T = T @ Ti
    return T


# ──────────────────────────────────────────────────────────────────
# CSV グループ検出
# ──────────────────────────────────────────────────────────────────

@dataclass
class StrokeGroup:
    """1ステーション分のストロークグループ（ユニーク接触点列）。"""
    pts: np.ndarray       # (M,3) ブレードローカル接触点（M>=2）
    normals: np.ndarray   # (M,3) ブレードローカル法線
    row_start: int        # 元CSVでのグループ先頭行 (0-based)
    row_end: int          # 元CSVでのグループ末尾行 (0-based, inclusive)


def load_blade_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """刃先CSV (x,y,z,nx,ny,nz) を読み込み (pts, normals) を返す。"""
    data = np.loadtxt(path, delimiter=",")
    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError(f"刃先CSVの形式が不正です（6列必要）: {path}")
    return data[:, :3].astype(float), data[:, 3:6].astype(float)


def detect_stroke_groups(pts: np.ndarray, normals: np.ndarray,
                         tol: float = 1e-9) -> List[StrokeGroup]:
    """連続重複行からストロークグループを汎用検出する。

    各グループは「重複ペアで始まり、次の重複ペアで終わる」行ブロック。
    ユニーク接触点はグループ先頭重複の2行目から末尾重複の1行目まで。
    """
    pts = np.asarray(pts, dtype=float)
    normals = np.asarray(normals, dtype=float)
    n = len(pts)
    rows = np.hstack([pts, normals])
    dup = [i for i in range(n - 1)
           if np.allclose(rows[i], rows[i + 1], atol=tol)]

    if len(dup) < 2 or len(dup) % 2 != 0:
        raise ValueError(
            f"刃先CSVのグループ構造を検出できません（重複ペア数={len(dup)}）")

    groups: List[StrokeGroup] = []
    expected_start = 0
    for k in range(0, len(dup), 2):
        s, e = dup[k], dup[k + 1]       # 重複ペア (s,s+1) で開始、(e,e+1) で終了
        if s != expected_start or e <= s + 1:
            raise ValueError(
                f"刃先CSVのグループ境界が不正です（行{s}〜{e}）")
        # ユニーク接触点: s+1 〜 e（先頭重複の2行目 〜 末尾重複の1行目）
        idx = list(range(s + 1, e + 1))
        groups.append(StrokeGroup(
            pts=pts[idx].copy(), normals=normals[idx].copy(),
            row_start=s, row_end=e + 1))
        expected_start = e + 2
    if expected_start != n:
        raise ValueError(
            f"刃先CSVの末尾にグループ外の行があります（行{expected_start}〜）")
    return groups


def split_sides(groups: List[StrokeGroup]
                ) -> Tuple[List[StrokeGroup], List[StrokeGroup]]:
    """法線X成分の符号で 左(nx<0) / 右(nx>0) に分割する。"""
    left  = [g for g in groups if float(np.mean(g.normals[:, 0])) < 0.0]
    right = [g for g in groups if float(np.mean(g.normals[:, 0])) >= 0.0]
    return left, right


# ──────────────────────────────────────────────────────────────────
# 姿勢計算（curve_follow と同じ規約）
# ──────────────────────────────────────────────────────────────────

def _rotation_between(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """v_from を v_to へ回す最小回転行列（単位ベクトル前提）。"""
    axis = np.cross(v_from, v_to)
    norm = float(np.linalg.norm(axis))
    dot = float(np.clip(np.dot(v_from, v_to), -1.0, 1.0))
    if norm < 1e-12:
        return np.eye(3)  # 平行（反平行はCSV内では発生しない想定）
    angle_deg = np.degrees(np.arctan2(norm, dot))
    return _rotation_about_axis(axis, angle_deg)


def _group_flange_Ts(g: StrokeGroup, t: np.ndarray,
                     T_blade_inv: np.ndarray,
                     C: np.ndarray, s: np.ndarray, x_stone: np.ndarray,
                     edge_angle_deg: float,
                     T_off_world: Optional[np.ndarray] = None
                     ) -> List[np.ndarray]:
    """1グループ分の各接触点に対するフランジ変換 T_ee（ワールド）を返す。

    先頭点の法線を curve_follow と同じ規約（法線を砥石法線と反平行に整列
    + 接線まわりの刃付け角）でアンカーし、以降の点はグループ内の
    相対法線回転を剛体的に適用する。これにより各点で
      R_k @ n_k = -s（法線は常に砥石法線と反平行）
    を厳密に保ちながら、グループ内の姿勢変化が砥石接線まわりの
    純回転（揺動）になる — 理想LSの W/R 一定・P のみ変化と同じパターン。
    """
    n0 = _normalize(g.normals[0])
    B_src = _orthonormal_basis(n0, t)
    B_tgt = _orthonormal_basis(-s, x_stone)
    R0 = B_tgt @ B_src.T
    if abs(edge_angle_deg) > 1e-9:
        t_world = R0 @ t
        R0 = _rotation_about_axis(t_world, edge_angle_deg) @ R0

    Ts: List[np.ndarray] = []
    for k in range(len(g.pts)):
        n_k = _normalize(g.normals[k])
        R_k = R0 @ _rotation_between(n_k, n0)  # R_k @ n_k = R0 @ n0 = -s
        T_bw = np.eye(4)
        T_bw[:3, :3] = R_k
        T_bw[:3, 3] = C - R_k @ g.pts[k]
        if T_off_world is not None:
            # パス→ツールオフセット: 接触フレーム（C アンカー）で共役した
            # ワールド変換を前置乗算 — 純回転なら接触点 C は不動。
            T_bw = T_off_world @ T_bw
        Ts.append(T_bw @ T_blade_inv)
    return Ts


def _make_wp(T_ee: np.ndarray, label: str, *,
             motion: MotionType = MotionType.LINEAR,
             speed: float = STROKE_SPEED,
             cnt: Optional[int] = CNT_VAL,
             joint_pct: Optional[int] = None) -> Waypoint:
    from ..robot.kinematics import Kinematics
    x, y, z, rx, ry, rz = Kinematics.transform_to_pose(T_ee)
    return Waypoint(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz,
                    speed=speed, motion_type=motion, label=label,
                    cnt=cnt, joint_speed_pct=joint_pct)


# ──────────────────────────────────────────────────────────────────
# プログラム生成
# ──────────────────────────────────────────────────────────────────

@dataclass
class KenmaPrograms:
    """生成結果: 3プログラム + 統計情報。"""
    route_left: Route          # HaL（左側面）
    route_right: Route         # HaR（右側面）
    route_main: Route          # kenma（CALL 構造のメイン）
    n_groups_left: int = 0
    n_groups_right: int = 0
    n_unreachable: int = 0
    unreachable_labels: List[str] = field(default_factory=list)
    uframe_pos: Tuple[float, ...] = ()
    utool_pos: Tuple[float, ...] = UTOOL9_POS
    q_left_start: Optional[np.ndarray] = None   # HaL 先頭ホバーの関節解 [rad]
    q_right_start: Optional[np.ndarray] = None  # HaR 先頭ホバーの関節解 [rad]


def _ik_quality(kin, T: np.ndarray, q: np.ndarray) -> Tuple[float, float]:
    """IK解の品質 (位置誤差mm, 回転行列フロベニウス誤差) を返す。"""
    Tc = kin.forward(q)
    perr = float(np.linalg.norm(Tc[:3, 3] - T[:3, 3]))
    rerr = float(np.linalg.norm(Tc[:3, :3] - T[:3, :3]))
    return perr, rerr


def _fold_revolute(q: np.ndarray, ref: np.ndarray,
                   lo: np.ndarray, up: np.ndarray) -> np.ndarray:
    """各関節を ±360° 折り返して ref に近づける（リミット内のみ）。"""
    q = q.copy()
    two_pi = 2.0 * np.pi
    for i in range(len(q)):
        for cand in (q[i] - two_pi, q[i] + two_pi):
            if (lo[i] <= cand <= up[i]
                    and abs(cand - ref[i]) < abs(q[i] - ref[i])):
                q[i] = cand
    return q


# 腰・肩側の関節を重視した距離（J1 が回り込む解を強く減点）
_IK_BRANCH_W = np.array([3.0, 2.0, 2.0, 1.0, 1.0, 0.5])


def _collect_ik_candidates(kin, T: np.ndarray, ref: np.ndarray,
                           rng: np.random.Generator,
                           pos_tol: float, rot_tol: float,
                           max_restarts: int, n_candidates: int,
                           dedupe: bool = False) -> List[np.ndarray]:
    """多点リスタート L-BFGS-B で品質ゲート済みIK候補を収集する。

    kin.inverse は1回あたり多数の内部リスタートを行い重いので、
    ここでは1スタートずつの L-BFGS-B を直接回す。
    dedupe=True のとき、±360°折り返し後 1° 丸めで同一視できる
    候補（同一ブランチ）を除外する。
    """
    from scipy.optimize import minimize
    lo, up = kin.dh.get_joint_limits()
    bounds = list(zip(lo, up))
    p_t, R_t = T[:3, 3], T[:3, :3]

    def cost(qv):
        Tc = kin.forward(qv)
        return (np.sum((Tc[:3, 3] - p_t) ** 2)
                + 100.0 * np.sum((Tc[:3, :3] - R_t) ** 2))

    candidates: List[np.ndarray] = []
    seen: set = set()
    for _ in range(max_restarts):
        q0 = rng.uniform(lo, up)
        res = minimize(cost, q0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
        perr, rerr = _ik_quality(kin, T, res.x)
        if perr < pos_tol and rerr < rot_tol:
            q = _fold_revolute(res.x, ref, lo, up)
            if dedupe:
                key = tuple(int(v) for v in np.round(np.degrees(q)))
                if key in seen:
                    continue
                seen.add(key)
            candidates.append(q)
            if len(candidates) >= n_candidates:
                break
    return candidates


def _robust_ik(kin, T: np.ndarray, q_seed: Optional[np.ndarray],
               rng: np.random.Generator,
               pos_tol: float = 0.5, rot_tol: float = 1e-2,
               max_restarts: int = 150, n_candidates: int = 6
               ) -> Tuple[Optional[np.ndarray], bool]:
    """品質ゲート付きIK。シード解で追跡し、外れたら多点リスタートで探索する。

    kin.inverse の成功判定は位置1mmのみで姿勢誤差を見ないため、
    ここで位置 pos_tol[mm]・回転 rot_tol（フロベニウス）の両方を検証する。
    グローバル探索時は複数候補からレディ姿勢に最も近い解を選ぶ
    （後ろ向きに回り込むような異常ブランチを避ける）。
    """
    if q_seed is not None:
        # シード自体が既に解（例: enumerate_ik_branches で選んだ
        # スタート関節）なら、そのブランチをそのまま採用する。
        perr, rerr = _ik_quality(kin, T, q_seed)
        if perr < pos_tol and rerr < rot_tol:
            return np.asarray(q_seed, dtype=float).copy(), True
        q, ok = kin.inverse(T, q_init=q_seed)
        if ok and q is not None:
            perr, rerr = _ik_quality(kin, T, q)
            if perr < pos_tol and rerr < rot_tol:
                return q, True

    # 多点リスタート（先頭点や追跡が外れた場合のグローバル探索）。
    ref = q_seed if q_seed is not None else kin.dh.ready_position()
    candidates = _collect_ik_candidates(kin, T, ref, rng,
                                        pos_tol, rot_tol,
                                        max_restarts, n_candidates)
    if not candidates:
        return None, False
    best = min(candidates,
               key=lambda qv: float(np.sum(_IK_BRANCH_W * (qv - ref) ** 2)))
    return best, True


def enumerate_ik_branches(kin, T: np.ndarray,
                          q_seed: Optional[np.ndarray] = None,
                          *,
                          pos_tol: float = 0.5, rot_tol: float = 1e-2,
                          max_restarts: int = 120, max_branches: int = 8,
                          rng: Optional[np.random.Generator] = None
                          ) -> List[np.ndarray]:
    """姿勢 T に対する IK 解ブランチ（候補関節配置）を列挙する。

    RoboDK の「スタート地点に好ましい関節」リスト相当:
      1. 品質ゲート（位置 pos_tol mm / 回転 rot_tol）を通る解のみ収集
      2. ±360° 折り返しでレディ姿勢（または q_seed）へ寄せる
      3. 1° 丸めで重複ブランチを除去
      4. レディ姿勢への加重距離が近い順にソート
      5. 最大 max_branches 件を返す

    Returns:
        List[np.ndarray] — 各要素は 6 関節角 [rad]。空リスト = 到達不能。
    """
    if rng is None:
        rng = np.random.default_rng(7)
    ref = (np.asarray(q_seed, dtype=float)
           if q_seed is not None else kin.dh.ready_position())

    branches: List[np.ndarray] = []
    seen: set = set()

    def _try_add(q: Optional[np.ndarray]):
        if q is None:
            return
        perr, rerr = _ik_quality(kin, T, q)
        if perr >= pos_tol or rerr >= rot_tol:
            return
        lo, up = kin.dh.get_joint_limits()
        qf = _fold_revolute(np.asarray(q, dtype=float), ref, lo, up)
        key = tuple(int(v) for v in np.round(np.degrees(qf)))
        if key in seen:
            return
        seen.add(key)
        branches.append(qf)

    # 解析解/シード追跡を最初の候補として試す（高速・最有力ブランチ）
    q0, ok = kin.inverse(T, q_init=ref)
    if ok:
        _try_add(q0)

    for q in _collect_ik_candidates(kin, T, ref, rng, pos_tol, rot_tol,
                                    max_restarts, max_branches * 3,
                                    dedupe=True):
        key = tuple(int(v) for v in np.round(np.degrees(q)))
        if key not in seen:
            seen.add(key)
            branches.append(q)
        if len(branches) >= max_branches * 2:
            break

    branches.sort(key=lambda qv: float(np.sum(_IK_BRANCH_W * (qv - ref) ** 2)))
    return branches[:max_branches]


def _side_waypoints(groups: List[StrokeGroup], side: str,
                    T_blade_inv: np.ndarray,
                    C: np.ndarray, s: np.ndarray, x_stone: np.ndarray,
                    edge_angle_deg: float, hover_mm: float,
                    T_off_world: Optional[np.ndarray] = None,
                    rev_flags: Optional[List[bool]] = None
                    ) -> List[Waypoint]:
    """1側面分のウェイポイント列（グループごとに hover/接触/ストローク/retract）。

    rev_flags[gi]=True のグループは姿勢列を前進時と同一に計算した上で
    トラバース順のみ逆転する（RoboDK の「逆方向」— 同じ物理接触を
    逆向きに掃引。ホバー/リトラクトは逆転後の先頭/末尾から導出）。
    """
    # グループ接線: グループ重心間の方向（刃渡り方向）。端は隣の値を流用。
    centroids = [g.pts.mean(axis=0) for g in groups]
    tangents: List[np.ndarray] = []
    for i in range(len(groups)):
        if len(groups) >= 2:
            j0, j1 = (i, i + 1) if i + 1 < len(groups) else (i - 1, i)
            t = _normalize(centroids[j1] - centroids[j0])
        else:
            t = np.array([0.0, 1.0, 0.0])
        if np.linalg.norm(t) < 1e-9:
            t = np.array([0.0, 1.0, 0.0])
        tangents.append(t)

    wps: List[Waypoint] = []
    for gi, g in enumerate(groups):
        t = tangents[gi]
        T_list = _group_flange_Ts(g, t, T_blade_inv,
                                  C, s, x_stone, edge_angle_deg,
                                  T_off_world)
        if rev_flags is not None and rev_flags[gi]:
            T_list = T_list[::-1]
        gname = f"{side}_G{gi + 1:02d}"

        # ホバー: 接触開始姿勢を砥石面法線方向に +hover_mm
        T_hover = T_list[0].copy()
        T_hover[:3, 3] = T_hover[:3, 3] + hover_mm * s
        if gi == 0:
            # 先頭グループのみ J 動作（理想LSの J P[1] 20% CNT1）
            wps.append(_make_wp(T_hover, f"{gname}_HOVER",
                                motion=MotionType.JOINT,
                                speed=100.0, joint_pct=JOINT_PCT))
        else:
            wps.append(_make_wp(T_hover, f"{gname}_HOVER",
                                speed=CONTACT_SPEED))

        # 接触開始
        wps.append(_make_wp(T_list[0], f"{gname}_CONTACT",
                            speed=CONTACT_SPEED))
        # ストローク点（接触開始以降のユニーク点）
        for k in range(1, len(T_list)):
            wps.append(_make_wp(T_list[k], f"{gname}_S{k}",
                                speed=STROKE_SPEED))
        # リトラクト: 最終接触姿勢 +hover_mm
        T_ret = T_list[-1].copy()
        T_ret[:3, 3] = T_ret[:3, 3] + hover_mm * s
        wps.append(_make_wp(T_ret, f"{gname}_RETRACT",
                            speed=CONTACT_SPEED))
    return wps


GroupSelection = Union[int, Tuple[int, bool]]


def _normalize_selection(
        selected_groups: Optional[Sequence[GroupSelection]],
        reversed_flags: Optional[Sequence[bool]],
        n_groups: int
) -> Tuple[Optional[List[int]], Optional[List[bool]]]:
    """selected_groups の int / (int, reversed) 混在を正規化する。"""
    if selected_groups is None:
        if reversed_flags is None:
            return None, None
        if len(reversed_flags) != n_groups:
            raise ValueError(
                f"reversed_flags の長さが不正です "
                f"({len(reversed_flags)} != グループ数 {n_groups})")
        return list(range(n_groups)), [bool(r) for r in reversed_flags]
    idxs: List[int] = []
    revs: List[bool] = []
    for item in selected_groups:
        if isinstance(item, (tuple, list)):
            idxs.append(int(item[0]))
            revs.append(bool(item[1]))
        else:
            idxs.append(int(item))
            revs.append(False)
    if reversed_flags is not None:
        if len(reversed_flags) != len(idxs):
            raise ValueError("reversed_flags と selected_groups の長さが一致しません")
        revs = [bool(r) for r in reversed_flags]
    return idxs, revs


def _prepare_sides(groups: List[StrokeGroup],
                   selected_groups: Optional[Sequence[GroupSelection]],
                   reversed_flags: Optional[Sequence[bool]]
                   ) -> Tuple[List[StrokeGroup], List[bool],
                              List[StrokeGroup], List[bool], str]:
    """選択・逆方向フラグを適用して左右グループ列＋フラグ列と先頭側を返す。

    Returns:
        (left, left_rev, right, right_rev, first_side)
    """
    idxs, revs = _normalize_selection(selected_groups, reversed_flags,
                                      len(groups))
    if idxs is not None:
        bad = [i for i in idxs if not (0 <= i < len(groups))]
        if bad:
            raise ValueError(f"曲線インデックスが範囲外です: {bad}")
        ordered = [(groups[i], r) for i, r in zip(idxs, revs)]
        # 左右へ振り分け（ユーザー指定の相対順を保持）
        left_p  = [(g, r) for g, r in ordered
                   if float(np.mean(g.normals[:, 0])) < 0.0]
        right_p = [(g, r) for g, r in ordered
                   if float(np.mean(g.normals[:, 0])) >= 0.0]
        if not left_p and not right_p:
            raise ValueError("曲線が選択されていません")
        first_side = ("L" if float(np.mean(ordered[0][0].normals[:, 0])) < 0.0
                      else "R")
        left,  left_rev  = ([g for g, _ in left_p],  [r for _, r in left_p])
        right, right_rev = ([g for g, _ in right_p], [r for _, r in right_p])
    else:
        left, right = split_sides(groups)
        if not left or not right:
            raise ValueError(
                f"左右側面の検出に失敗しました（左={len(left)} / 右={len(right)}グループ）")
        left_rev  = [False] * len(left)
        right_rev = [False] * len(right)
        first_side = "L"
    return left, left_rev, right, right_rev, first_side


def _tool_offset_world(tool_offset: Optional[np.ndarray],
                       T_contact: np.ndarray) -> Optional[np.ndarray]:
    """パス→ツールオフセット式を接触フレーム共役のワールド変換にする。

    T_c = 接触フレーム回転を接触点 C にアンカーした 4x4 とし、
      M = T_c @ T_expr @ T_c^-1
    を返す。純回転の T_expr では接触点 C が不動のまま姿勢のみ変わり、
    transl 成分は接触フレーム軸（X=砥石接線/刃渡り方向, Z=砥石面法線）
    に沿って接触点を平行移動する。None / 恒等行列は None を返す
    （従来生成とバイト同一の出力を保証）。
    """
    if tool_offset is None:
        return None
    T_expr = np.asarray(tool_offset, dtype=float)
    if T_expr.shape != (4, 4):
        raise ValueError("tool_offset は 4x4 行列が必要です")
    if np.array_equal(T_expr, np.eye(4)):
        return None
    Tc = np.eye(4)
    Tc[:3, :3] = T_contact[:3, :3]
    Tc[:3, 3] = T_contact[:3, 3]
    return Tc @ T_expr @ np.linalg.inv(Tc)


def first_hover_T(
    blade_pts: np.ndarray,
    blade_normals: np.ndarray,
    T_blade: np.ndarray,
    T_contact: np.ndarray,
    *,
    edge_angle_deg: float = 0.0,
    hover_mm: float = HOVER_MM,
    selected_groups: Optional[Sequence[GroupSelection]] = None,
    reversed_flags: Optional[Sequence[bool]] = None,
    tool_offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    """先頭に選択された曲線が属する側の先頭ホバー姿勢（ワールド 4x4）を返す。

    generate_kenma_programs と同一の数式で計算するため、ここで得た姿勢の
    IK 解は生成時の先頭ウェイポイント（P[1]/HOME）のシードとして使える。
    """
    groups = detect_stroke_groups(blade_pts, blade_normals)
    left, left_rev, right, right_rev, first_side = _prepare_sides(
        groups, selected_groups, reversed_flags)
    if first_side == "L" and left:
        side_groups, side_rev, side = left, left_rev, "L"
    else:
        side_groups, side_rev, side = right, right_rev, "R"

    C = T_contact[:3, 3]
    s = _normalize(T_contact[:3, 2])
    x_stone = T_contact[:3, 0]
    T_blade_inv = np.linalg.inv(T_blade)
    T_off = _tool_offset_world(tool_offset, T_contact)

    wps = _side_waypoints(side_groups, side, T_blade_inv, C, s, x_stone,
                          edge_angle_deg, hover_mm, T_off, side_rev)
    return wps[0].to_transform()


def generate_kenma_programs(
    blade_pts: np.ndarray,
    blade_normals: np.ndarray,
    T_blade: np.ndarray,
    T_contact: np.ndarray,
    kin=None,
    *,
    edge_angle_deg: float = 0.0,
    hover_mm: float = HOVER_MM,
    prog_left: str = "HaL",
    prog_right: str = "HaR",
    prog_main: str = "kenma",
    selected_groups: Optional[Sequence[GroupSelection]] = None,
    reversed_flags: Optional[Sequence[bool]] = None,
    tool_offset: Optional[np.ndarray] = None,
    q_start: Optional[np.ndarray] = None,
) -> KenmaPrograms:
    """刃先CSV点列から kenma 形式 3 プログラム（HaL/HaR/kenma）を生成する。

    Args:
        blade_pts / blade_normals : 刃先CSV（ブレードローカル）
        T_blade   : フランジ → ブレードローカル取付オフセット (4x4)
        T_contact : ワールドでの砥石接触フレーム（UF9 STONE, +Z=砥石面法線）
        kin       : Kinematics（IK 到達性検証に使用、None でスキップ）
        edge_angle_deg : 接線まわりの追加刃付け角（CSV法線が揺動を含むため通常0）
        hover_mm  : ホバー/リトラクト離隔距離
        selected_groups : 使用するストロークグループの選択列
                          （detect_stroke_groups の返り値に対する添字・実行順）。
                          各要素は int または (int, reversed: bool) タプル。
                          reversed=True のグループは接触点順を逆転して掃引する
                          （RoboDK の「逆方向」相当 — 同じ物理接触を逆向きに）。
                          None = 全グループを既定順で使用（従来動作）。
                          左側面の選択は HaL へ・右側面は HaR へ、それぞれ
                          指定の相対順を保って振り分けられる。片側が空の場合
                          その側のプログラムは空になり、kenma メインの CALL
                          も省略される。
        reversed_flags : selected_groups と平行な逆方向フラグ列（省略可）。
                         指定時はタプル内のフラグより優先される。
        tool_offset : パス→ツールへの追加オフセット（4x4, transl mm/rot）。
                      parse_pose_expression の結果を渡す。接触フレーム
                      （X=砥石接線/刃渡り方向＝刃付け角度軸, Z=砥石面法線）
                      で共役され、純回転では接触点が不動のまま姿勢のみ変わる。
                      None / 恒等行列 = 従来どおり（出力はバイト同一）。
        q_start : スタート地点に好ましい関節 [rad]（enumerate_ik_branches の
                  選択結果）。各側面の先頭ホバーIKのシードとなり、そのまま
                  P[1]/HOME の関節解として LS に出力される。None = 従来どおり
                  グローバル探索。
    Returns:
        KenmaPrograms
    """
    from ..robot.kinematics import Kinematics

    groups = detect_stroke_groups(blade_pts, blade_normals)
    left, left_rev, right, right_rev, _first_side = _prepare_sides(
        groups, selected_groups, reversed_flags)

    C = T_contact[:3, 3]
    s = _normalize(T_contact[:3, 2])
    x_stone = T_contact[:3, 0]
    T_blade_inv = np.linalg.inv(T_blade)
    T_off = _tool_offset_world(tool_offset, T_contact)

    wps_left = _side_waypoints(left, "L", T_blade_inv, C, s, x_stone,
                               edge_angle_deg, hover_mm, T_off, left_rev)
    wps_right = _side_waypoints(right, "R", T_blade_inv, C, s, x_stone,
                                edge_angle_deg, hover_mm, T_off, right_rev)

    # IK 連続性チェック（直前解をシードに順次解く・品質ゲート付き）
    n_unreachable = 0
    unreachable_labels: List[str] = []
    q_left_start: Optional[np.ndarray] = None
    q_right_start: Optional[np.ndarray] = None
    if kin is not None:
        rng = np.random.default_rng(7)
        for side_wps in (wps_left, wps_right):
            # 各側面の先頭: q_start 指定時はそれをシードに、
            # 未指定時は従来どおりグローバル探索から
            q_seed = (np.asarray(q_start, dtype=float).copy()
                      if q_start is not None else None)
            for i, wp in enumerate(side_wps):
                q, ok = _robust_ik(kin, wp.to_transform(), q_seed, rng)
                if ok and q is not None:
                    q_seed = q
                    if i == 0:
                        if side_wps is wps_left:
                            q_left_start = q.copy()
                        else:
                            q_right_start = q.copy()
                else:
                    n_unreachable += 1
                    unreachable_labels.append(wp.label)

    uframe_pos = tuple(round(float(v), 6)
                       for v in Kinematics.transform_to_pose(T_contact))

    route_left = Route(name=prog_left, comment="RoboDK sequence",
                       uframe=UF_NUM, utool=UT_NUM)
    route_left.waypoints = wps_left
    route_right = Route(name=prog_right, comment="RoboDK sequence",
                        uframe=UF_NUM, utool=UT_NUM)
    route_right.waypoints = wps_right

    # メインプログラム kenma:
    #   J ホーム → CALL HaL → J ホーム → J HaR側安全姿勢 → CALL HaR →
    #   J 安全姿勢 → J ホーム
    # ホーム / 安全姿勢 = 各側面の先頭ホバー姿勢（エクスポート時に IK で関節化）
    def _joint_wp(src: Waypoint, label: str) -> Waypoint:
        wp = copy.deepcopy(src)
        wp.label = label
        wp.motion_type = MotionType.JOINT
        wp.speed = 100.0
        wp.joint_speed_pct = JOINT_PCT
        wp.cnt = None  # FINE（理想 kenma.LS と同じ）
        return wp

    def _call_wp(prog: str) -> Waypoint:
        return Waypoint(call=prog, label=f"CALL {prog}",
                        motion_type=MotionType.JOINT)

    route_main = Route(name=prog_main, comment="RoboDK sequence",
                       uframe=UF_NUM, utool=UT_NUM)
    main_wps: List[Waypoint] = []
    if wps_left:
        main_wps += [
            _joint_wp(wps_left[0],  "HOME"),
            _call_wp(prog_left),
            _joint_wp(wps_left[0],  "HOME"),
        ]
    if wps_right:
        main_wps += [
            _joint_wp(wps_right[0], "SAFE_R"),
            _call_wp(prog_right),
            _joint_wp(wps_right[0], "SAFE_R"),
        ]
    if wps_left:
        main_wps.append(_joint_wp(wps_left[0], "HOME"))
    route_main.waypoints = main_wps

    return KenmaPrograms(
        route_left=route_left,
        route_right=route_right,
        route_main=route_main,
        n_groups_left=len(left),
        n_groups_right=len(right),
        n_unreachable=n_unreachable,
        unreachable_labels=unreachable_labels,
        uframe_pos=uframe_pos,
        utool_pos=UTOOL9_POS,
        q_left_start=q_left_start,
        q_right_start=q_right_start,
    )


def build_playback_sequence(result: KenmaPrograms) -> List[Waypoint]:
    """kenma メインの CALL をインライン展開した再生用ウェイポイント列を返す。"""
    seq: List[Waypoint] = []
    inline = {
        result.route_left.name:  result.route_left.waypoints,
        result.route_right.name: result.route_right.waypoints,
    }
    for wp in result.route_main.waypoints:
        if wp.call is not None:
            seq.extend(copy.deepcopy(inline.get(wp.call, [])))
        else:
            seq.append(copy.deepcopy(wp))
    return seq


def export_kenma_ls(result: KenmaPrograms, out_dir: str, kin=None,
                    mn_comments: Optional[List[str]] = None) -> List[str]:
    """3プログラムを out_dir に kenma.LS / HaL.LS / HaR.LS として出力する。

    Returns:
        出力ファイルパスのリスト [kenma, HaL, HaR]。
    """
    import os
    from .tp_exporter import TPExporter

    exporter = TPExporter(kin)
    paths: List[str] = []

    # HaL / HaR: UF/UT の PR 定義 + デカルト /POS + CNT1
    # 先頭の J ポイントには生成時の IK 解（品質ゲート済み）を使用する。
    side_solutions = (
        {0: result.q_left_start} if result.q_left_start is not None else None,
        {0: result.q_right_start} if result.q_right_start is not None else None,
    )
    for route, sols in zip((result.route_left, result.route_right),
                           side_solutions):
        if not route.waypoints:
            continue   # 片側のみ選択時: 空プログラムは出力しない
        path = os.path.join(out_dir, f"{route.name}.LS")
        exporter.export(route, path,
                        uframe_pos=result.uframe_pos,
                        utool_pos=result.utool_pos,
                        cart_pos=True, ideal_attr=True,
                        preserve_name_case=True,
                        mn_comments=mn_comments,
                        joint_solutions=sols)
        paths.append(path)

    # kenma メイン: 関節位置のみ + CALL（理想 kenma.LS 同様 PR 定義なし）
    # ホーム = HaL 先頭ホバー解 / 安全姿勢 = HaR 先頭ホバー解
    # メイン構造は選択内容で可変のためラベルから index を解決する
    main_solutions = {}
    for i, wp in enumerate(result.route_main.waypoints):
        if wp.call is not None:
            continue
        if wp.label == "HOME" and result.q_left_start is not None:
            main_solutions[i] = result.q_left_start
        elif wp.label == "SAFE_R" and result.q_right_start is not None:
            main_solutions[i] = result.q_right_start
    main_solutions = main_solutions or None
    main_path = os.path.join(out_dir, f"{result.route_main.name}.LS")
    exporter.export(result.route_main, main_path,
                    cart_pos=True, ideal_attr=True,
                    preserve_name_case=True,
                    mn_comments=["Generated by robot_sim",
                                 "kenma main sequence"],
                    joint_solutions=main_solutions)
    paths.insert(0, main_path)
    return paths
