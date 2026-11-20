@echo off
cd /d "%~dp0"
echo === Starting JJSimon sweep at %date% %time% > both_sweeps.log
python -u tmp_jjsimon_v2_sweep.py >> both_sweeps.log 2>&1
if errorlevel 1 (
    echo === JJSimon sweep FAILED with exit %errorlevel% >> both_sweeps.log
    exit /b %errorlevel%
)
echo === JJSimon sweep DONE at %date% %time% >> both_sweeps.log
if exist account_mgmt_checkpoint.pkl del account_mgmt_checkpoint.pkl
echo === Starting account mgmt sweep at %date% %time% >> both_sweeps.log
python -u tmp_account_mgmt_sweep.py >> both_sweeps.log 2>&1
echo === ALL DONE at %date% %time% >> both_sweeps.log
