# デスクトップにショートカットを作成するスクリプト
# 使い方: リポジトリフォルダ内で PowerShell から実行してください
#   .\create_shortcut.ps1

$RepoDir    = Split-Path -Parent $MyInvocation.MyCommand.Path
$BatchFile  = Join-Path $RepoDir "launch.bat"
$Desktop    = [Environment]::GetFolderPath("Desktop")
$Shortcut   = Join-Path $Desktop "刃付けロボットシミュレータ.lnk"

$WshShell   = New-Object -ComObject WScript.Shell
$Lnk        = $WshShell.CreateShortcut($Shortcut)

$Lnk.TargetPath       = $BatchFile
$Lnk.WorkingDirectory = $RepoDir
$Lnk.WindowStyle      = 1   # 通常ウィンドウ
$Lnk.Description      = "FANUC LR Mate 200iD/14L 刃付けシミュレータ"

# Python アイコンが見つかれば使用、なければデフォルト
$PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if ($PythonExe) {
    $Lnk.IconLocation = "$PythonExe,0"
}

$Lnk.Save()

Write-Host "ショートカットを作成しました:" -ForegroundColor Green
Write-Host "  $Shortcut" -ForegroundColor Cyan
