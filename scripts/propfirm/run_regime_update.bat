@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM run_regime_update.bat
REM ─────────────────────────────────────────────────────────────────────────
REM Launcher for the daily HMM regime update. Designed to be called by
REM Windows Task Scheduler at ~17:15 ET each trading day (after CME daily
REM session settlement at 17:00 ET).
REM
REM What it does:
REM   1. cd into the repo root
REM   2. run scripts\propfirm\regime_update.py
REM   3. write a timestamped log line to scripts\propfirm\regime_update.log
REM   4. exit with the script's exit code
REM
REM Schedule via:
REM   schtasks /Create /TN "PropfirmRegimeUpdate" ^
REM     /TR "D:\Personal_Projects\Backtest Engine\scripts\propfirm\run_regime_update.bat" ^
REM     /SC DAILY /ST 17:15 /F
REM ─────────────────────────────────────────────────────────────────────────

setlocal

REM Resolve repo root from this script's location (scripts\propfirm\..\..)
set "REPO_ROOT=%~dp0..\.."
pushd "%REPO_ROOT%"

set "LOG=%~dp0regime_update.log"
set "PY=C:\Python312\python.exe"

echo. >> "%LOG%"
echo === %DATE% %TIME% === >> "%LOG%"

"%PY%" "%~dp0regime_update.py" >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"

if %RC% NEQ 0 (
    echo FAILED with exit code %RC% >> "%LOG%"
) else (
    echo OK >> "%LOG%"
)

popd
endlocal & exit /b %RC%
