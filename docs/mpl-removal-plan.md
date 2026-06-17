# matplotlib(CPU) ビューポート版 削除計画

> ステータス: **実施済み（v0.21.0）** — 記録日 2026-06-17 / 実施日 2026-06-17
> 決定: matplotlib(CPU) 版 3D ビューポートを削除し、VisPy(OpenGL/GPU) 版に一本化した。
> GPU 初期化失敗時はフォールバックせず、エラーダイアログを表示してアプリを終了する。

## 背景

3D ビューは長らく matplotlib(CPU) 版 `Viewport3D` を使ってきたが、
v0.20.0 で VisPy(OpenGL/GPU) 版 `ViewportGPU` を既定に昇格した。
現在は両バックエンドを `viewport_factory.create_viewport()` で選択しており、
GPU 初期化に失敗すると自動で mpl 版へフォールバックしてアプリが必ず起動する。

両方の実機（Intel UHD 620 / metal2022）で GPU 版の動作を確認済みのため、
mpl 版を保守対象から外して GPU 単独構成にする。

## 決定事項

- **GPU 単独構成にする**（mpl 版・フォールバックを削除）。
- **GPU 初期化失敗時はフォールバックせず、分かりやすいエラーを表示して起動を中止する。**
  - 例: 「3D ビューの初期化に失敗しました。GPU/OpenGL ドライバを確認してください。」
    といったメッセージ＋例外内容を表示し、`sys.exit(1)` 等で終了する。
  - 黙ってクラッシュさせず、何が原因か（GPU 不可）が利用者に伝わるようにする。

## 削除・変更対象

| 対象 | 内容 |
| --- | --- |
| `robot_sim/gui/viewport.py` (約1564行) | mpl 版本体 `Viewport3D` を**ファイルごと削除**。 |
| `robot_sim/gui/viewport_factory.py` | バックエンド選択・フォールバックを削除。`create_viewport()` は GPU 版を生成し、失敗時はエラー表示して終了する薄いラッパに簡素化（または `main_window` から `ViewportGPU` を直接生成）。 |
| `robot_sim/gui/main_window.py` | `self._is_gpu` 分岐（10箇所以上）を GPU 前提で整理。下記参照。 |
| `requirements.txt` | matplotlib が GPU 版で本当に不要か確認のうえ整理（※ render_frame/動画保存で matplotlib に依存していないか要確認。`_FigShim` 経由のため不要の見込みだが要検証）。 |
| `README.md` / changelog | GPU 必須である旨を明記。`ROBOT_VIEWPORT` 環境変数の説明を削除/更新。 |

### `main_window.py` の `_is_gpu` 分岐（GPU 前提で整理）

`_is_gpu` は常に True 相当になるため、mpl 側のコードパスを削除して分岐を畳む。

- L756 `self._is_gpu = getattr(self.viewport, "realtime", False)` → 不要化（常時 GPU）。
- L757-759 ビューポートラベル `[GPU/OpenGL]` → 無条件表示。
- L1572-1588 「滑らか再生」ボタンのラベル/ツールチップ → GPU 版（リアルタイム再生）のみ残す。
- L1589-1607 FPS スピン（事前描画用）→ **削除**（mpl 専用 UI）。
- L1624-1638 軽量表示トグル/自動軽量化 → **削除**（mpl 専用 UI）。
- L3544 / L3562 ボタンラベル分岐 → GPU 文言に固定。
- L3604 / L4014-4019 / L4212 リアルタイム再生委譲・事前描画スキップ → GPU 前提で簡素化。

### 事前描画・動画保存パスの注意

- `_prerender_frames()`（L4032〜）は **GPU 版の動画保存でも使用される**。
  `self.viewport.fig`（GPU 版では `_FigShim`）と `render_frame()` 経由で動作するため、
  この関数自体は**残す**。mpl 専用の「滑らか再生＝事前描画」用途のみ削除する。
- GPU 版は `viewport_gpu.py` に `realtime=True` / `fig=_FigShim` / `render_frame()` を
  実装済み（L179 / L229 / L869）。動画オンデマンド capture はここに依存。

## 作業手順（段階的に実施推奨）

1. **factory の簡素化**: フォールバックを撤去し、GPU 初期化失敗時はエラー表示＋終了に変更。
   （この時点で mpl 版コードはまだ残すが、到達不能になる。動作確認）
2. **main_window の分岐整理**: `_is_gpu` を撤去し、mpl 専用 UI（FPS スピン・軽量表示トグル）と
   mpl 用コードパスを削除。GPU 文言に固定。
3. **viewport.py 削除**: `Viewport3D` をファイルごと削除。残存 import が無いことを確認。
4. **依存整理**: matplotlib が他で使われていないか確認のうえ `requirements.txt` を更新。
5. **ドキュメント更新**: README / changelog に GPU 必須を明記、`ROBOT_VIEWPORT` 記述を更新。

## 影響・リスク

- **CPU フォールバックが無くなる**: GPU/OpenGL が使えない環境ではアプリが起動できなくなる
  （決定通り。エラー表示で利用者に原因が伝わるようにする）。
- 既存の実機 2 台は GPU 動作確認済み。新規環境に展開する際は
  `pip install vispy pyopengl pyopengltk` と GPU/ドライバが前提になる旨を周知する。

## 検証チェックリスト（実施時）

- [ ] 通常起動（GPU 利用可）で従来通り動作する。
- [ ] GPU 強制失敗時（例: 環境変数や擬似的に import 失敗）に、分かりやすいエラーが出て終了する。
- [ ] リアルタイム再生・クリックピッキング・ルート表示が動く。
- [ ] 動画保存（MP4/GIF）が GPU 版で正常に出力される。
- [ ] `viewport.py` 削除後、未解決 import / 参照が無い（grep で確認）。
