@echo off
setlocal
cd /d "%~dp0.."
set LOG=out\gemini_run.log
echo ==== run started %date% %time% ==== > "%LOG%"

REM --- locate conda ---
where conda >nul 2>nul
if %errorlevel%==0 goto condafound
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" call "%USERPROFILE%\anaconda3\Scripts\activate.bat" & goto condafound
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" call "%USERPROFILE%\miniconda3\Scripts\activate.bat" & goto condafound
if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" call "C:\ProgramData\anaconda3\Scripts\activate.bat" & goto condafound
echo Conda not found. Open "Anaconda Prompt", cd to this folder, and run the .bat there.
echo Conda not found >> "%LOG%"
pause
exit /b 1

:condafound
echo conda located >> "%LOG%"
call conda env list | findstr /C:"vlm-advisor" >nul
if %errorlevel%==0 goto envexists
echo [1/3] Creating conda env "vlm-advisor" - python 3.11 ...
call conda create -y -n vlm-advisor python=3.11 >> "%LOG%" 2>&1
if not %errorlevel%==0 goto fail

:envexists
call conda activate vlm-advisor
if not %errorlevel%==0 goto fail
echo [2/3] Installing dependencies...
pip install -r requirements.txt >> "%LOG%" 2>&1
if not %errorlevel%==0 goto fail

echo [3/3] Running pipeline with Gemini - 16 API calls...
python -m src.pipeline --input "out/warehouse_v3_0001-0384.mp4" --provider gemini --out out/annotated_gemini.mp4 --max-samples 16 2>> "%LOG%"
if not %errorlevel%==0 goto fail

echo.
echo DONE. Outputs: out\annotated_gemini.mp4 + out\annotated_gemini_telemetry.jsonl
echo run finished OK >> "%LOG%"
pause
exit /b 0

:fail
echo.
echo Something failed - see out\gemini_run.log for details.
pause
exit /b 1
