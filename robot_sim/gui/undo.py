"""
スナップショット方式の Undo/Redo マネージャ。

MainWindow から capture() / restore() コールバックを受け取り、
「操作直前の全状態スナップショット」をスタックに積む方式。
個々の操作に逆操作を実装する必要がなく、あらゆる機能を一律に
やり直し可能にできる。

使い方:
    mgr = UndoManager(capture=self._capture_state,
                      restore=self._restore_state)
    mgr.push("マーカー追加")      # 変更を加える「直前」に呼ぶ
    ...状態を変更...
    mgr.undo()                    # 直前の操作を取り消す
    mgr.redo()                    # 取り消した操作をやり直す

coalesce=True を指定すると、同一ラベルの連続操作（マウスホイール連打・
スライダードラッグ等）が一定時間内に続く限り 1 回の undo 単位に
まとめられる。
"""
from __future__ import annotations

import time
from typing import Any, Callable, List, Optional, Tuple


class UndoManager:
    """ラベル付きスナップショットの undo / redo スタック。"""

    def __init__(self, capture: Callable[[], Any],
                 restore: Callable[[Any], None],
                 limit: int = 50):
        self._capture = capture
        self._restore = restore
        self._limit = limit
        self._undo_stack: List[Tuple[str, Any]] = []
        self._redo_stack: List[Tuple[str, Any]] = []
        self._last_label: Optional[str] = None
        self._last_time = 0.0

    # ── 記録 ──────────────────────────────────────────────────────────

    def push(self, label: str, coalesce: bool = False,
             coalesce_sec: float = 1.5):
        """変更を加える直前に呼び、現在の状態をスナップショットする。

        coalesce=True: 直前の push と同一ラベルかつ coalesce_sec 秒以内
        なら新しいスナップショットを積まない（連続操作をまとめる）。
        """
        now = time.time()
        if (coalesce and self._undo_stack
                and self._last_label == label
                and now - self._last_time < coalesce_sec):
            self._last_time = now
            self._redo_stack.clear()
            return
        self._undo_stack.append((label, self._capture()))
        if len(self._undo_stack) > self._limit:
            del self._undo_stack[0]
        self._redo_stack.clear()
        self._last_label = label
        self._last_time = now

    # ── 取り消し / やり直し ────────────────────────────────────────────

    def undo(self) -> Optional[str]:
        """直前の操作を取り消す。戻り値は操作ラベル（なければ None）。"""
        if not self._undo_stack:
            return None
        label, snap = self._undo_stack.pop()
        self._redo_stack.append((label, self._capture()))
        self._restore(snap)
        self._last_label = None   # 取り消し後の連続操作はまとめない
        return label

    def redo(self) -> Optional[str]:
        """取り消した操作をやり直す。戻り値は操作ラベル（なければ None）。"""
        if not self._redo_stack:
            return None
        label, snap = self._redo_stack.pop()
        self._undo_stack.append((label, self._capture()))
        self._restore(snap)
        self._last_label = None
        return label

    # ── 状態問い合わせ（メニュー表示用） ────────────────────────────────

    @property
    def undo_label(self) -> Optional[str]:
        return self._undo_stack[-1][0] if self._undo_stack else None

    @property
    def redo_label(self) -> Optional[str]:
        return self._redo_stack[-1][0] if self._redo_stack else None
