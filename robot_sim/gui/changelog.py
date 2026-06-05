"""Version and changelog dialog for the FANUC LR Mate 200iD/14L simulator."""
import tkinter as tk
from tkinter import ttk

APP_VERSION = "0.3.9"

# (version, date, time, changes)
CHANGELOG = [
    ("0.3.9", "2026-06-05", "03:00", [
        "TCP・ターゲット管理パネルを追加（右パネル上部）",
        "「+ TCP」ボタン: 現在のロボットTCP位置にTCPマーカー（シアン★）を追加",
        "「+ 🎯」ボタン: 現在のTCP位置にターゲットマーカー（オレンジ）を追加",
        "「削除」ボタン: リスト選択マーカーを削除",
        "「現在TCP→」ボタン: ロボット現在TCP座標をX/Y/Z入力欄に自動入力",
        "X/Y/Z入力 + 「適用」で選択マーカーの位置を自由に調整",
        "3Dビューポートにマーカーをリアルタイム表示",
    ]),
    ("0.3.8", "2026-06-05", "01:00", [
        "右ドラッグズームを無効化（すべてのボタン操作イベントをブロック）",
        "ビューポートラベルを更新: ホイール拡大縮小・STL/CSVドロップ説明を追加",
        "STL/CSV ドラッグアンドドロップ対応: numpy-stl でSTL読込、CSV点群表示",
        "オーバーレイ位置姿勢コントロールパネルを追加 (X/Y/Z/Rx/Ry/Rz 入力・適用・クリア)",
        "requirements.txt 更新: numpy-stl・tkinterdnd2 追加、pybullet・open3d 削除",
    ]),
    ("0.3.7", "2026-06-05", "00:35", [
        "ロボット外観を v0.3.5 の形状に戻した（円柱ベース・黒胴体）",
    ]),
    ("0.3.6", "2026-06-04", "23:45", [
        "ロボット外観を大幅改善（実機FANUC LR Mate 200iD/14Lに近づけた）",
        "ベース台座を黄色の大型ボックスに変更（ワイヤーフレームから実体へ）",
        "J1胴体を黒から黄色の太いコラム＋肩ハウジングに変更",
        "上腕・前腕を矩形断面ボックスリンクに変更（円柱→角柱）",
        "_box_link・_rotated_box ヘルパーを新規追加",
    ]),
    ("0.3.5", "2026-06-04", "23:15", [
        "3Dビュー左右の余白を最小化（軸ラベル・目盛りを非表示）",
        "デフォルトズームを調整してロボットが画面中央に表示",
        "視点角度を最適化（elev=25°, azim=-45°）",
    ]),
    ("0.3.4", "2026-06-04", "23:00", [
        "3Dビューポートの左ドラッグ回転を無効化（混乱防止）",
        "格子（グリッド）表示を削除してビューをすっきり",
        "3Dビューポートを画面いっぱいに拡大（余白・ツールバーを除去）",
        "ビューポート幅比率を拡大（5:2）",
    ]),
    ("0.3.3", "2026-06-04", "22:30", [
        "関節スライダーを縦向きから横向きに変更（よりコンパクトに）",
        "ツールチップ追加: マウスカーソルを当てると各機能の説明が表示",
        "ロボット3D表示を円柱・球体で実際の見た目に近づけた",
        "ベース・関節・リンクをFANUC黄色で立体的に描画",
        "ジョグ軸ラベルにJoint/Cartesianの日本語説明を追加",
    ]),
    ("0.3.2", "2026-06-04", "21:30", [
        "UI デザインを全面刷新: GitHub Dark テーマ、カラーコード統一",
        "各パネルに日本語説明文を追加（ジョグ操作・IK・速度OVR・UTool/UFrame）",
        "ジョグ軸ラベルを Joint/Cartesian モードに応じて切替表示",
        "シミュレーション進捗を % 表示で確認可能に",
        "FK 結果を X/Y/Z・Rx/Ry/Rz の見やすいフォーマットで表示",
        "ステータスバー右端にロボット名とバージョンを常時表示",
        "ダイアログに説明文を追加 (UTool/UFrame 編集・自動生成)",
    ]),
    ("0.3.1", "2026-06-04", "20:30", [
        "更新履歴に更新時刻 (時:分) を追加",
        "launch.bat 改良: 起動時に自動で最新版を取得 (git pull)",
        "刃付けルート自動生成ダイアログを追加 (ルートメニュー)",
        "更新履歴パネルをメイン画面右側に常時表示",
    ]),
    ("0.3.0", "2026-06-04", "14:00", [
        "ファナック LR Mate 200iD/14L の正式仕様を全面反映",
        "DH パラメータ修正: a1=50, a2=440, a3=35, d4=420mm (ROS-Industrial URDF 準拠)",
        "関節最大速度を追加: J1-J6 各軸の°/s 制限を管理",
        "ツールフレーム (UTool) 定義機能を追加",
        "ユーザーフレーム (UFrame) 定義機能を追加",
        "速度オーバーライド (Speed Override) スライダーを追加",
        "ジョグ操作 (Cartesian / Joint) パネルを追加",
        "3D ビューポート改善: フロアグリッド・ワークスペース可視化・TCP 表示",
        "チェンジログ画面を追加 (ヘルプメニュー)",
        "ロボット仕様ダイアログを追加",
    ]),
    ("0.2.0", "2026-06-04", "10:30", [
        "DH パラメータを Z-up 標準座標系に修正 (Y-up → Z-up)",
        "マウスホイールでの 3D ズーム機能を追加",
        "視点ドラッグ後の回転角度を保持するよう修正",
        "レディポジション: J2=-45° J3=+30° J5=-60° に設定",
        "デフォルト刃付けルートの座標系を統一",
    ]),
    ("0.1.0", "2026-06-04", "09:00", [
        "初期実装: FANUC LR Mate 200iD/14L シミュレータ",
        "Modified DH 順・逆運動学 (解析解 + 数値フォールバック)",
        "matplotlib 3D ビューポート",
        "ウェイポイント編集 (追加/編集/削除)",
        "CSV インポート/エクスポート",
        "FANUC TP (.ls) プログラムエクスポート",
    ]),
]


def show_changelog(parent):
    """Show changelog dialog."""
    win = tk.Toplevel(parent)
    win.title(f"バージョン情報 / Changelog — v{APP_VERSION}")
    win.geometry("720x540")
    win.configure(bg="#1A1A1A")

    hdr = tk.Label(
        win,
        text=f"FANUC LR Mate 200iD/14L  刃付けシミュレータ  v{APP_VERSION}",
        bg="#1A1A1A", fg="#F5C400", font=("", 12, "bold"),
    )
    hdr.pack(pady=(12, 4))
    sub = tk.Label(
        win, text="Knife Sharpening Robot Simulator",
        bg="#1A1A1A", fg="#888888", font=("", 9),
    )
    sub.pack()

    ttk.Separator(win).pack(fill=tk.X, padx=12, pady=8)

    frame = tk.Frame(win, bg="#1A1A1A")
    frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
    sb = tk.Scrollbar(frame)
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    txt = tk.Text(
        frame, bg="#111111", fg="#CCCCCC", font=("Courier", 9),
        yscrollcommand=sb.set, wrap=tk.WORD, state="normal",
        borderwidth=0, highlightthickness=0,
    )
    txt.pack(fill=tk.BOTH, expand=True)
    sb.config(command=txt.yview)

    for ver, date, time_, items in CHANGELOG:
        txt.insert(tk.END, f"v{ver}  {date}  {time_}\n", "ver")
        for item in items:
            txt.insert(tk.END, f"  • {item}\n", "item")
        txt.insert(tk.END, "\n")

    txt.tag_config("ver", foreground="#F5C400", font=("Courier", 10, "bold"))
    txt.tag_config("item", foreground="#CCCCCC")
    txt.config(state="disabled")

    ttk.Button(win, text="閉じる (Close)", command=win.destroy).pack(pady=8)
