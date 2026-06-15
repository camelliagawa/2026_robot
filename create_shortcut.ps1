# Create desktop shortcut for the Blade Sharpening Robot Simulator
# Usage: run this script from the repository folder
#   .\create_shortcut.ps1

$RepoDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$LaunchPs1 = Join-Path $RepoDir "launch.ps1"
$Desktop   = [Environment]::GetFolderPath("Desktop")
$Shortcut  = Join-Path $Desktop "Simulator.lnk"

# Find pythonw (no console) or fall back to python
$pyCmd = Get-Command pythonw -ErrorAction SilentlyContinue
if (-not $pyCmd) {
    $pyCmd = Get-Command python -ErrorAction SilentlyContinue
}
$PythonExe = if ($pyCmd) { $pyCmd.Source } else { $null }

$WshShell = New-Object -ComObject WScript.Shell
$Lnk = $WshShell.CreateShortcut($Shortcut)
$Lnk.TargetPath       = "powershell.exe"
$Lnk.Arguments        = "-WindowStyle Hidden -ExecutionPolicy Bypass -File `"$LaunchPs1`""
$Lnk.WorkingDirectory = $RepoDir
$Lnk.WindowStyle      = 1
$Lnk.Description      = "FANUC LR Mate 200iD/14L Simulator"

if ($PythonExe) {
    $Lnk.IconLocation = "$PythonExe,0"
}

$Lnk.Save()
Write-Host "Shortcut created: $Shortcut"
