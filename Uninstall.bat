@echo off
rem removes kimodo motion from houdini, and the folder too if it only holds the tool's files
setlocal
cd /d "%~dp0"
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "PY=%~dp0.deps\python\cpython-3.10.22-windows-x86_64-none\python.exe"
set "PYTHONHOME="
set "PYTHONPATH="

echo Removing Kimodo Motion for Houdini (%ROOT%)
echo Close Houdini first: the Kimodo service it started ends with it.
echo.
if not exist "%PY%" (
    echo Install.bat did not finish here, so Houdini was never linked to this folder. Without the Python it
    echo installs, what the folder holds cannot be checked, so it is not offered for deletion.
    goto :kept
)
rem exit code 3: the folder holds other things too, keep it
"%PY%" -E -s -u -X utf8 install\install.py --uninstall
if errorlevel 4 goto :failed
if errorlevel 3 goto :kept
if errorlevel 1 goto :failed

echo.
choice /c YN /m "Also delete %ROOT% and everything in it, including the downloaded models"
if errorlevel 2 goto :kept
rem end the batch file first so it can delete its own folder, then cd out of it
(goto) 2>nul & cd /d "%ROOT%\.." & rd /s /q "%ROOT%" & (if exist "%ROOT%" (echo Some files could not be deleted; close Houdini and delete "%ROOT%" by hand.) else (echo Deleted "%ROOT%".)) & pause

:kept
echo.
echo The folder was kept: %ROOT%
pause
exit /b 0

:failed
echo.
echo The package pointer could not be removed; the reason is above.
pause
exit /b 1
