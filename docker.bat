@echo off
setlocal

rem Get repo root
for %%I in ("%~dp0.") do set "ROOT=%%~fI"

rem Default to baseline if no argument
if "%1"=="" (
    set "METHOD=baseline"
) else (
    set "METHOD=%1"
)

echo Checking Docker...
docker info >nul 2>&1 || (
    echo ERROR: Docker not running. Start Docker Desktop and retry.
    exit /b 1
)

if not exist "%ROOT%\data\raw" (
    echo ERROR: Missing %ROOT%\data\raw
    echo Unzip Kaggle data into data\raw\ and retry.
    exit /b 1
)

if not exist "%ROOT%\submissions" mkdir "%ROOT%\submissions"

echo Building image...
cd /d "%ROOT%"
docker build -q -t classicml -f docker\Dockerfile . || exit /b 1

echo Running method: %METHOD%
docker run --rm -e MODE=%METHOD% -v "%ROOT%\data\raw:/app/data/raw" -v "%ROOT%\submissions:/app/submissions" classicml

endlocal

