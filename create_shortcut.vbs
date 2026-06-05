Set objShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")

strScriptDir = objFSO.GetParentFolderName(WScript.ScriptFullName)
strDesktop   = objShell.SpecialFolders("Desktop")
strShortcut  = strDesktop & "\刃付けロボットシミュレータ.lnk"

Set oLink = objShell.CreateShortcut(strShortcut)
oLink.TargetPath       = strScriptDir & "\launch.bat"
oLink.WorkingDirectory = strScriptDir
oLink.Description      = "Blade Sharpening Robot Simulator"
oLink.WindowStyle      = 1
oLink.Save

MsgBox "デスクトップにショートカットを作成しました。" & vbCrLf & strShortcut, 64, "完了"
