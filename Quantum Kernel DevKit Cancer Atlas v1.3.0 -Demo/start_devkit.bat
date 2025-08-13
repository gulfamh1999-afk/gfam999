@echo off
setlocal

REM Always run from this folder (works when double-clicked)
cd /d "%~dp0"

REM Choose Python (prefer py 3.10, fallback to python)
where py >nul 2>nul && (set "PY=py -3.10") || (set "PY=python")

REM Create venv if missing
if not exist ".venv" (
  %PY% -m venv .venv
)

REM Activate venv
call .venv\Scripts\activate

REM Install deps (first run takes a few minutes)
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Make sure Streamlit is NOT headless so it can open the browser
set STREAMLIT_SERVER_HEADLESS=false
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

REM Start the server in a separate window so this script can continue
set PORT=8501
start "DevKit Server" cmd /c ".venv\Scripts\python -m streamlit run app.py --server.port %PORT% --server.headless false"

REM Give the server a moment to boot, then open the browser
timeout /t 4 >nul
start "" "http://localhost:%PORT%"

echo.
echo DevKit launching at http://localhost:%PORT%
echo If the tab doesn't open, copy/paste the URL above into your browser.
echo (If another app is using port %PORT%, change the PORT value in this .bat.)
