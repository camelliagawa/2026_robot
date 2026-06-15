# デスクトップにショートカットを作成するスクリプト
# 使い方: リポジトリフォルダ内で PowerShell から実行してください
#   .\create_shortcut.ps1

$RepoDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$LaunchPs1 = Join-Path $RepoDir "launch.ps1"
$Desktop   = [Environment]::GetFolderPath("Desktop")
$Shortcut  = Join-Path $Desktop "刃付けロボットシミュレータ.lnk"

# アイコン用に pythonw / python を探す
$PythonExe = (Get-Command pythonw -ErrorAction SilentlyContinue)?.Source
if (-not $PythonExe) {
    $PythonExe = (Get-Command python -ErrorAction SilentlyContinue)?.Source
}

$WshShell = New-Object -ComObject WScript.Shell
$Lnk = $WshShell.CreateShortcut($Shortcut)

# PowerShell を非表示ウィンドウで起動 → コンソール不要
$Lnk.TargetPath       = "powershell.exe"
$Lnk.Arguments        = "-WindowStyle Hidden -ExecutionPolicy Bypass -File `"$LaunchPs1`""
$Lnk.WorkingDirectory = $RepoDir
$Lnk.WindowStyle      = 1   # 通常ウィンドウ（アプリ自体は通常表示）
$Lnk.Description      = "FANUC LR Mate 200iD/14L 刃付けシミュレータ"

if ($PythonExe) {
    $Lnk.IconLocation = "$PythonExe,0"
}

$Lnk.Save()

Write-Host "Shortcut created/updated:" -ForegroundColor Green
Write-Host "  $Shortcut" -ForegroundColor Cyan
Write-Host "  -> コンソールウィンドウなしで起動します" -ForegroundColor Cyan
