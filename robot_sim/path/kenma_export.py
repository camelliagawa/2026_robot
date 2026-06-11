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
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
                     edge_angle_deg: float) -> List[np.ndarray]:
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
        q, ok = kin.inverse(T, q_init=q_seed)
        if ok and q is not None:
            perr, rerr = _ik_quality(kin, T, q)
            if perr < pos_tol and rerr < rot_tol:
                return q, True

    # 多点リスタート（先頭点や追跡が外れた場合のグローバル探索）。
    # kin.inverse は1回あたり多数の内部リスタートを行い重いので、
    # ここでは1スタートずつの L-BFGS-B を直接回す。
    from scipy.optimize import minimize
    lo, up = kin.dh.get_joint_limits()
    bounds = list(zip(lo, up))
    p_t, R_t = T[:3, 3], T[:3, :3]
    ref = q_seed if q_seed is not None else kin.dh.ready_position()
    # 腰・肩側の関節を重視した距離（J1 が回り込む解を強く減点）
    w = np.array([3.0, 2.0, 2.0, 1.0, 1.0, 0.5])

    def cost(qv):
        Tc = kin.forward(qv)
        return (np.sum((Tc[:3, 3] - p_t) ** 2)
                + 100.0 * np.sum((Tc[:3, :3] - R_t) ** 2))

    candidates: List[np.ndarray] = []
    for _ in range(max_restarts):
        q0 = rng.uniform(lo, up)
        res = minimize(cost, q0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
        perr, rerr = _ik_quality(kin, T, res.x)
        if perr < pos_tol and rerr < rot_tol:
            candidates.append(_fold_revolute(res.x, ref, lo, up))
            if len(candidates) >= n_candidates:
                break
    if not candidates:
        return None, False
    best = min(candidates, key=lambda qv: float(np.sum(w * (qv - ref) ** 2)))
    return best, True


def _side_waypoints(groups: List[StrokeGroup], side: str,
                    T_blade_inv: np.ndarray,
                    C: np.ndarray, s: np.ndarray, x_stone: np.ndarray,
                    edge_angle_deg: float, hover_mm: float) -> List[Waypoint]:
    """1側面分のウェイポイント列（グループごとに hover/接触/ストローク/retract）。"""
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
                                  C, s, x_stone, edge_angle_deg)
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
) -> KenmaPrograms:
    """刃先CSV点列から kenma 形式 3 プログラム（HaL/HaR/kenma）を生成する。

    Args:
        blade_pts / blade_normals : 刃先CSV（ブレードローカル）
        T_blade   : フランジ → ブレードローカル取付オフセット (4x4)
        T_contact : ワールドでの砥石接触フレーム（UF9 STONE, +Z=砥石面法線）
        kin       : Kinematics（IK 到達性検証に使用、None でスキップ）
        edge_angle_deg : 接線まわりの追加刃付け角（CSV法線が揺動を含むため通常0）
        hover_mm  : ホバー/リトラクト離隔距離
    Returns:
        KenmaPrograms
    """
    from ..robot.kinematics import Kinematics

    groups = detect_stroke_groups(blade_pts, blade_normals)
    left, right = split_sides(groups)
    if not left or not right:
        raise ValueError(
            f"左右側面の検出に失敗しました（左={len(left)} / 右={len(right)}グループ）")

    C = T_contact[:3, 3]
    s = _normalize(T_contact[:3, 2])
    x_stone = T_contact[:3, 0]
    T_blade_inv = np.linalg.inv(T_blade)

    wps_left = _side_waypoints(left, "L", T_blade_inv, C, s, x_stone,
                               edge_angle_deg, hover_mm)
    wps_right = _side_waypoints(right, "R", T_blade_inv, C, s, x_stone,
                                edge_angle_deg, hover_mm)

    # IK 連続性チェック（直前解をシードに順次解く・品質ゲート付き）
    n_unreachable = 0
    unreachable_labels: List[str] = []
    q_left_start: Optional[np.ndarray] = None
    q_right_start: Optional[np.ndarray] = None
    if kin is not None:
        rng = np.random.default_rng(7)
        for side_wps in (wps_left, wps_right):
            q_seed = None  # 各側面の先頭はグローバル探索から
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
    route_main.waypoints = [
        _joint_wp(wps_left[0],  "HOME"),
        _call_wp(prog_left),
        _joint_wp(wps_left[0],  "HOME"),
        _joint_wp(wps_right[0], "SAFE_R"),
        _call_wp(prog_right),
        _joint_wp(wps_right[0], "SAFE_R"),
        _joint_wp(wps_left[0],  "HOME"),
    ]

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
    main_solutions = None
    if result.q_left_start is not None and result.q_right_start is not None:
        qL, qR = result.q_left_start, result.q_right_start
        main_solutions = {0: qL, 2: qL, 3: qR, 5: qR, 6: qL}
    main_path = os.path.join(out_dir, f"{result.route_main.name}.LS")
    exporter.export(result.route_main, main_path,
                    cart_pos=True, ideal_attr=True,
                    preserve_name_case=True,
                    mn_comments=["Generated by robot_sim",
                                 "kenma main sequence"],
                    joint_solutions=main_solutions)
    paths.insert(0, main_path)
    return paths
