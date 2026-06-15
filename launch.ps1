# Blade Sharpening Robot Simulator - silent launcher
# Called by the desktop shortcut (no console window)
Set-Location $PSScriptRoot

# Auto-update (ignore errors)
try { git pull 2>&1 | Out-Null } catch {}

# pythonw = no console window; fall back to python if not found
$pyCmd = Get-Command pythonw -ErrorAction SilentlyContinue
if (-not $pyCmd) {
    $pyCmd = Get-Command python -ErrorAction SilentlyContinue
}

if ($pyCmd) {
    & $pyCmd.Source -m robot_sim.main
} else {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show(
        "Python not found. Please install Python and try again.",
        "Launch Error",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Error)
}
