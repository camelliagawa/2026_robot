"""
設定ファイル (JSON) の入出力 — RoboDK のステーション保存に相当する軽量版。

アプリの主要な設定値をまとめて 1 つの JSON ファイルへ保存し、
「ファイル」メニューまたは 3D ビューポートへのドラッグ＆ドロップで
まとめて復元できるようにする。

保存対象:
  ・関節角度 / アクティブ UTool / UFrame
  ・ツールフレーム (UT) / ユーザーフレーム (UF) の定義
  ・TCP/ターゲットマーカー・参照フレーム
  ・STL / CSV / 刃先CSV オーバーレイ（ファイルパス + 位置姿勢）
  ・経路点リスト（Route 全体）
  ・速度オーバーライド・軽量表示・ルート生成オフセット式
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np

from ..robot.kinematics import Kinematics
from ..robot.tool_frame import ToolFrame
from ..robot.user_frame import UserFrame
from ..path.route import Waypoint, MotionType

SETTINGS_MAGIC = "robot_sim_settings"

_FRAME_KEYS = ("number", "name", "x", "y", "z", "rx", "ry", "rz", "comment")
_WP_KEYS = ("x", "y", "z", "rx", "ry", "rz", "speed", "label",
            "cnt", "joint_speed_pct", "call")


def _frame_to_dict(f) -> dict:
    return {k: getattr(f, k) for k in _FRAME_KEYS}


def _wp_to_dict(wp: Waypoint) -> dict:
    d = {k: getattr(wp, k) for k in _WP_KEYS}
    d["motion_type"] = wp.motion_type.value
    return d


def _wp_from_dict(d: dict) -> Waypoint:
    kwargs = {k: d[k] for k in _WP_KEYS if k in d}
    kwargs["motion_type"] = MotionType(d.get("motion_type", "L"))
    return Waypoint(**kwargs)


# ──────────────────────────────────────────────────────────────────────
# 保存
# ──────────────────────────────────────────────────────────────────────

def collect_settings(mw) -> dict:
    """MainWindow の現在の状態を JSON 化可能な辞書として返す。"""
    from .changelog import APP_VERSION
    vp = mw.viewport

    ref_frames = []
    for rf in vp.get_ref_frames():
        pose = Kinematics.transform_to_pose(rf["T"])
        ref_frames.append({
            "name": rf["name"],
            "pose": [float(v) for v in pose],
            "color": rf.get("color", "#FF88FF"),
        })

    return {
        "app": SETTINGS_MAGIC,
        "version": APP_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "joint_angles_deg": [float(v) for v in np.rad2deg(mw._joint_angles)],
        "active_utool": mw._active_tool.number,
        "active_uframe": mw._active_uframe.number,
        "tool_frames": [_frame_to_dict(t) for t in mw.TOOL_FRAMES],
        "user_frames": [_frame_to_dict(u) for u in mw.USER_FRAMES],
        "markers": [dict(m) for m in mw._mk_list],
        "ref_frames": ref_frames,
        "stl": {
            "path": vp._stl_path,
            "pose": [float(v.get() or 0) for v in mw._stl_pose_vars],
        },
        "csv_overlay": {
            "path": vp._csv_path,
            "pose": [float(v.get() or 0) for v in mw._csv_pose_vars],
        },
        "blade": {
            "path": vp._blade_path,
            "pose": [float(v.get() or 0) for v in mw._blade_pose_vars],
        },
        "route": {
            "name": mw.route.name,
            "comment": mw.route.comment,
            "uframe": mw.route.uframe,
            "utool": mw.route.utool,
            "waypoints": [_wp_to_dict(wp) for wp in mw.route.waypoints],
        },
        "speed_override": int(mw._speed_override.get()),
        "fast_mode": bool(mw._fast_mode_var.get()),
        "auto_fast_playback": bool(mw._auto_fast_var.get()),
        "kenma_tool_offset": mw._kenma_dlg_state.get(
            "offset", "rotx(0)*roty(0)*rotz(0)"),
    }


def save_settings(mw, path: str):
    """現在の設定を JSON ファイルへ書き出す。"""
    data = collect_settings(mw)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────────────
# 読み込み
# ──────────────────────────────────────────────────────────────────────

def load_settings(path: str) -> dict:
    """設定 JSON を読み込んで検証する。形式不正なら ValueError。"""
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict) or data.get("app") != SETTINGS_MAGIC:
        raise ValueError(
            "このアプリの設定ファイルではありません"
            f"（\"app\": \"{SETTINGS_MAGIC}\" がありません）")
    return data


def is_settings_file(path: str) -> bool:
    """ドラッグ＆ドロップされた JSON が設定ファイルか判定する。"""
    try:
        load_settings(path)
        return True
    except (ValueError, OSError, json.JSONDecodeError):
        return False


def apply_settings(mw, data: dict) -> list:
    """設定辞書を MainWindow へ反映する。戻り値は警告メッセージのリスト。

    呼び出し側が事前に Undo スナップショットを積むこと。
    """
    warns: list = []
    vp = mw.viewport

    # ── フレーム定義 ──────────────────────────────────────────────
    tfs = [ToolFrame(**d) for d in data.get("tool_frames", [])]
    if tfs:
        mw.TOOL_FRAMES[:] = tfs
    ufs = [UserFrame(**d) for d in data.get("user_frames", [])]
    if ufs:
        mw.USER_FRAMES[:] = ufs

    mw._utool_combo.config(values=[
        f"UT{t.number}: {t.name}  (z={t.z:.0f}mm)" for t in mw.TOOL_FRAMES])
    mw._uframe_combo.config(values=[
        f"UF{u.number}: {u.name}" for u in mw.USER_FRAMES])

    ti = next((i for i, t in enumerate(mw.TOOL_FRAMES)
               if t.number == data.get("active_utool")), 0)
    mw._utool_combo.current(ti)
    mw._active_tool = mw.TOOL_FRAMES[ti]
    vp.set_tool_frame(mw._active_tool)

    ui = next((i for i, u in enumerate(mw.USER_FRAMES)
               if u.number == data.get("active_uframe")), 0)
    mw._uframe_combo.current(ui)
    mw._active_uframe = mw.USER_FRAMES[ui]
    vp.set_user_frame(mw._active_uframe)

    # ── 参照フレーム（UF9 は _sync_uf9 が正規の姿勢で再登録する） ──
    if "ref_frames" in data:
        vp.clear_ref_frames()
        for rf in data["ref_frames"]:
            vp.add_ref_frame(rf["name"], *rf["pose"],
                             color=rf.get("color", "#FF88FF"))
    uf9 = next((u for u in mw.USER_FRAMES if u.number == 9), None)
    if uf9 is not None:
        mw._sync_uf9(uf9)
    mw._rf_refresh_listbox()

    # ── マーカー ──────────────────────────────────────────────────
    if "markers" in data:
        mw._mk_list = [dict(m) for m in data["markers"]]
        mw._mk_tcp_count = sum(1 for m in mw._mk_list if m["type"] == "tcp")
        mw._mk_tgt_count = len(mw._mk_list) - mw._mk_tcp_count
        mw._mk_refresh_listbox()
        mw._mk_sync_viewport()

    # ── オーバーレイ（STL / CSV / 刃先CSV） ──────────────────────
    def _apply_layer(key, pose_vars, load, apply_pose, clear):
        layer = data.get(key)
        if layer is None:
            return
        path = layer.get("path") or ""
        if path:
            if os.path.isfile(path):
                if not load(path):
                    warns.append(f"{key}: 読込失敗 ({path})")
            else:
                warns.append(f"{key}: ファイルが見つかりません ({path})")
        else:
            clear()
        pose = layer.get("pose")
        if pose and len(pose) >= 6:
            for var, val in zip(pose_vars, pose):
                var.set(f"{float(val):.2f}")
            apply_pose()

    _apply_layer("stl", mw._stl_pose_vars, vp.load_stl,
                 mw._apply_stl_pose, vp.clear_stl)
    _apply_layer("csv_overlay", mw._csv_pose_vars, vp.load_csv_points,
                 mw._apply_csv_pose, vp.clear_csv)

    blade = data.get("blade")
    if blade is not None:
        pose = blade.get("pose")
        if pose and len(pose) >= 6:
            for var, val in zip(mw._blade_pose_vars, pose):
                var.set(f"{float(val):.2f}")
        path = blade.get("path") or ""
        if path:
            if os.path.isfile(path):
                # _load_blade_csv が現在の取付オフセット（↑で設定済）を適用する
                mw._load_blade_csv(path)
            else:
                warns.append(f"刃先CSV: ファイルが見つかりません ({path})")
        else:
            vp.clear_blade()
            mw._blade_csv_path = None

    # ── 経路 ──────────────────────────────────────────────────────
    r = data.get("route")
    if r is not None:
        mw.route.waypoints = [_wp_from_dict(d) for d in r.get("waypoints", [])]
        mw.route.name    = r.get("name", mw.route.name)
        mw.route.comment = r.get("comment", mw.route.comment)
        mw.route.uframe  = r.get("uframe", mw.route.uframe)
        mw.route.utool   = r.get("utool", mw.route.utool)
        mw.route_editor.set_route(mw.route)
        vp.set_route(mw.route)
        mw._invalidate_sim_solutions()

    # ── その他 ────────────────────────────────────────────────────
    if "speed_override" in data:
        mw._speed_override.set(int(data["speed_override"]))
    if "fast_mode" in data:
        mw._fast_mode_var.set(bool(data["fast_mode"]))
        vp.set_fast_mode(bool(data["fast_mode"]))
    if "auto_fast_playback" in data:
        mw._auto_fast_var.set(bool(data["auto_fast_playback"]))
    if "kenma_tool_offset" in data:
        mw._kenma_dlg_state["offset"] = str(data["kenma_tool_offset"])

    q = data.get("joint_angles_deg")
    if q and len(q) == 6:
        mw._set_angles(np.deg2rad(np.asarray(q, dtype=float)))

    if hasattr(mw, "_tree"):
        mw._tree_refresh()
    return warns
