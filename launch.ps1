# 刃付けロボットシミュレータ — サイレント起動スクリプト
# create_shortcut.ps1 が作るショートカットから呼ばれる（コンソールなし）
# 手動実行: powershell -ExecutionPolicy Bypass -File launch.ps1

Set-Location $PSScriptRoot

# 最新版を自動取得（エラーは無視して続行）
try { git pull 2>&1 | Out-Null } catch {}

# pythonw = tkinter GUI は表示されるがコンソールウィンドウが出ない
$py = (Get-Command pythonw -ErrorAction SilentlyContinue)?.Source
if (-not $py) {
    $py = (Get-Command python  -ErrorAction SilentlyContinue)?.Source
}

if ($py) {
    & $py -m robot_sim.main
} else {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show(
        "Python が見つかりません。`nPython をインストールしてから再度お試しください。",
        "起動エラー", [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Error)
}
