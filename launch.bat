@echo off
cd /d "%~dp0"

echo 最新版を確認中...
git pull origin claude/zen-ptolemy-tUCsq
if %ERRORLEVEL% neq 0 (
    echo.
    echo [警告] 更新の取得に失敗しました。オフラインの可能性があります。
    echo 現在インストール済みのバージョンで起動します。
    echo.
)

python -m robot_sim.main
if %ERRORLEVEL% neq 0 (
    echo.
    echo エラーが発生しました。ライブラリが未インストールの場合は以下を実行してください:
    echo   pip install -r requirements.txt
    pause
)
