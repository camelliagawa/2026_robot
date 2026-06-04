@echo off
cd /d "%~dp0"
python -m robot_sim.main
if %ERRORLEVEL% neq 0 (
    echo.
    echo エラーが発生しました。ライブラリが未インストールの場合は以下を実行してください:
    echo   pip install -r requirements.txt
    pause
)
