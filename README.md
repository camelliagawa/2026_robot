# FANUC LR Mate 200iD/14L 刃付けロボットシミュレータ

FANUC LR Mate 200iD/14L ロボットアームに包丁（エンドエフェクタ）を取り付けた  
**刃付け作業自動化**のためのロボットシミュレーションソフトウェアです。

---

## 概要

本ソフトウェアは以下の機能を提供します：

- **順運動学（FK）**: DH パラメータに基づく変換行列計算
- **逆運動学（IK）**: 解析解 + 数値解フォールバック（scipy 最適化）
- **3D ビジュアライゼーション**: matplotlib による対話型 3D ビューポート
- **経路計画**: GUI による 3D ウェイポイント設定 / CSV インポート・エクスポート
- **FANUC TP エクスポート**: `.ls` 形式のプログラム出力
- **PyBullet シミュレーション**: 物理シミュレーションエンジン統合

---

## ロボット仕様

| 項目 | 値 |
|------|-----|
| モデル | FANUC LR Mate 200iD/14L |
| 可搬重量 | 14 kg |
| リーチ | 911 mm |
| 繰り返し精度 | ±0.02 mm |
| 自由度 | 6 |

### DH パラメータ（Modified DH 記法）

| ジョイント | a (mm) | α (deg) | d (mm) | θオフセット (deg) | 可動範囲 |
|-----------|--------|---------|-------|--------------|---------|
| J1 | 75 | -90 | 330 | 0 | ±170° |
| J2 | 450 | 0 | 0 | -90 | -85°/+145° |
| J3 | 75 | -90 | 0 | 0 | -175°/+255° |
| J4 | 0 | 90 | 450 | 0 | ±190° |
| J5 | 0 | -90 | 0 | 0 | ±135° |
| J6 | 0 | 0 | 80 | 0 | ±360° |

---

## インストール

```bash
# 依存パッケージのインストール
pip install -r requirements.txt
```

### 必要なパッケージ

- Python 3.9+
- numpy >= 1.24
- scipy >= 1.10
- pybullet >= 3.2.5
- open3d >= 0.17
- matplotlib >= 3.7

---

## 使い方

### GUI の起動

```bash
# プロジェクトルートから実行
python -m robot_sim.main

# または直接
python robot_sim/main.py
```

### GUI 操作

1. **3D ビューポート**
   - マウスドラッグ: 視点回転
   - スクロール: ズーム
   - 右ドラッグ: パン

2. **ジョイントスライダー**
   - J1〜J6 を直接操作してロボットの姿勢を変更

3. **経路編集**
   - 「追加 (Add)」: 新規ウェイポイントを追加
   - 「編集 (Edit)」: 選択したウェイポイントを編集
   - 「削除 (Del)」: ウェイポイントを削除
   - 「サンプル (Sample)」: デモ用刃付け経路を読み込む

4. **シミュレーション実行**
   - 「▶ シミュレーション実行」ボタンで経路を再生

5. **TP エクスポート**
   - 「TP エクスポート」で FANUC `.ls` ファイルを出力

---

## CSV フォーマット

ウェイポイントを CSV ファイルでインポート・エクスポートできます。

```csv
# Route: KNIFE_SHARPEN
# Comment: Knife sharpening route
x_mm,y_mm,z_mm,rx_deg,ry_deg,rz_deg,speed_mmps,motion_type,label
300,0,450,180,0,0,30,J,Home
300,-100,300,180,15,0,50,L,Approach
300,-50,280,180,15,0,30,L,Stroke1
300,0,280,180,15,0,30,L,Stroke2
300,50,280,180,15,0,30,L,Stroke3
300,100,280,180,15,0,30,L,Stroke4
300,0,450,180,0,0,50,L,Retract
```

| 列 | 説明 |
|----|------|
| x_mm, y_mm, z_mm | ベースフレーム基準の位置（mm） |
| rx_deg, ry_deg, rz_deg | ZYX オイラー角（度） |
| speed_mmps | 動作速度（mm/s）または % |
| motion_type | `J`（関節補間）、`L`（直線補間）、`C`（円弧補間） |
| label | 任意のラベル名 |

---

## FANUC TP エクスポート形式

```
/PROG  KNIFE_SHARPEN
/ATTR
OWNER          = MNEDITOR;
COMMENT        = "Knife sharpening route";
/MN
   1:  UFRAME_NUM=0 ;
   2:  UTOOL_NUM=1 ;
   3:J P[1] 30% FINE    ;
   4:L P[2] 50mm/sec FINE    ;
   5:L P[3] 30mm/sec FINE    ;
/POS
P[1]{
   GP1:
	UF : 0, UT : 1,  ; Home
	J1 =    0.000 deg,	J2 =  -45.000 deg,	J3 =   45.000 deg,
	J4 =    0.000 deg,	J5 =  -90.000 deg,	J6 =    0.000 deg
};
/END
```

---

## プロジェクト構成

```
robot_sim/
├── __init__.py
├── main.py                 # エントリポイント（tkinter GUI）
├── robot/
│   ├── __init__.py
│   ├── dh_params.py        # DH パラメータ定義
│   ├── kinematics.py       # 順・逆運動学
│   └── urdf/
│       └── lrmate200id14l.urdf  # URDF モデル
├── simulation/
│   ├── __init__.py
│   └── simulator.py        # PyBullet シミュレーション管理
├── path/
│   ├── __init__.py
│   ├── route.py            # 経路データモデル
│   ├── csv_io.py           # CSV 入出力
│   └── tp_exporter.py      # FANUC TP エクスポータ
└── gui/
    ├── __init__.py
    ├── main_window.py      # メインウィンドウ
    ├── viewport.py         # 3D ビューポート
    └── route_editor.py     # 経路エディタパネル
requirements.txt
README.md
```

---

## 逆運動学について

本ソフトウェアは 2 段階の IK ソルバを使用しています：

1. **解析解（優先）**: 球状手首を持つ 6R ロボットの閉形式解
   - 手首中心を EE 姿勢から計算
   - J1, J2, J3 を三角形法（余弦定理）で解く
   - J4, J5, J6 を ZYZ オイラー角分解で解く

2. **数値解（フォールバック）**: scipy の L-BFGS-B 法
   - 位置・姿勢誤差を最小化
   - 複数初期値からのランダム再スタート

---

## 包丁ツール座標系

- ツール Z 軸: 刃長方向（フランジから約 200mm）
- ハンドル: 150mm
- 刃: 200mm（3mm 厚、45mm 幅）
- TCP（ツール先端点）: 刃先

---

## ライセンス

MIT License

---

*本ソフトウェアは研究・教育目的のシミュレーションです。  
実際のロボット操作には必ず安全確認と専門家の監督のもとで行ってください。*
