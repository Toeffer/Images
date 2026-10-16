@echo off
rem Startet Bildwerk unter Windows (Doppelklick genuegt).
rem Beim ersten Start wird eine virtuelle Python-Umgebung (.venv) angelegt und alles installiert.
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto check
echo Richte Bildwerk ein (einmalig, dauert etwa eine Minute) ...
py -3 -m venv .venv 2>nul
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if not exist ".venv\Scripts\python.exe" goto nopython

:check
if not exist ".venv\.installed" goto install
powershell -NoProfile -Command "exit [int]((Get-Item 'pyproject.toml').LastWriteTime -gt (Get-Item '.venv\.installed').LastWriteTime)"
if errorlevel 1 goto install
goto run

:install
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 goto failed
type nul > ".venv\.installed"

:run
".venv\Scripts\python.exe" -m bildwerk %*
if errorlevel 1 pause
exit /b

:nopython
echo.
echo Python wurde nicht gefunden.
echo Bitte Python 3.10 oder neuer von https://www.python.org/downloads/ installieren
echo (dabei "Add python.exe to PATH" anhaken) und start.bat erneut starten.
pause
exit /b 1

:failed
echo.
echo Die Installation ist fehlgeschlagen. Bitte die Meldungen oben pruefen.
pause
exit /b 1
