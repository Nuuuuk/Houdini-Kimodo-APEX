@echo off
rem installs kimodo motion: double-click it. safe to run again, finished steps are skipped
setlocal
cd /d "%~dp0"

set "UV_VERSION=0.13.10"
set "UV_SHA256=297b3881fe1adfb922becc2c1613202abb41dc83ea5ff43d472aa087ca1f8683"
set "PY_VERSION=3.10.22"

set "UV=%~dp0.deps\uv\uv.exe"
set "UV_CACHE_DIR=%~dp0.deps\uv-cache"
set "UV_PYTHON_INSTALL_DIR=%~dp0.deps\python"
set "UV_NO_CONFIG=1"
set "PY=%UV_PYTHON_INSTALL_DIR%\cpython-%PY_VERSION%-windows-x86_64-none\python.exe"
set "PYTHONHOME="
set "PYTHONPATH="

rem what goes where, before anything is downloaded
echo Kimodo Motion for Houdini will be installed into this folder:
echo   %~dp0
echo.
echo   .deps\    uv and Python 3.10, used only by this tool (0.1 GB)
echo   .venv\    the Python packages of the Kimodo service, with PyTorch (5 GB)
echo   models\   Meta-Llama-3-8B-Instruct, the text encoder's base model (15 GB),
echo             the Kimodo model with the text encoder's adapters (1.5 GB)
echo.
echo and packages\kimodo_motion.json will be written in your Houdini user prefs folder.
echo Press any key to start, or close this window to cancel.
pause >nul
echo.

if exist "%UV%" goto :python
echo uv
if not exist ".deps\uv" mkdir ".deps\uv"
curl.exe -fL -# --retry 3 -o ".deps\uv.zip" "https://github.com/astral-sh/uv/releases/download/%UV_VERSION%/uv-x86_64-pc-windows-msvc.zip" || goto :failed
rem check uv's checksum
certutil -hashfile ".deps\uv.zip" SHA256 | findstr /i /x "%UV_SHA256%" >nul || (
    echo The downloaded uv does not have the expected checksum.
    del ".deps\uv.zip"
    goto :failed
)
tar.exe -xf ".deps\uv.zip" -C ".deps\uv" || goto :failed
del ".deps\uv.zip"

:python
if exist "%PY%" goto :install
echo Python %PY_VERSION%
"%UV%" python install %PY_VERSION% --no-bin --no-registry -q || goto :failed

:install
"%PY%" -E -s -u -X utf8 install\install.py %* || goto :failed
echo.
pause
exit /b 0

:failed
echo.
echo The installation did not finish; the reason is above. Run Install.bat again once it is fixed.
pause
exit /b 1
