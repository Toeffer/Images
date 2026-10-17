@echo off
rem Startet Bildwerk unter Windows (Doppelklick genuegt).
rem Beim ersten Start wird eine virtuelle Python-Umgebung (.venv) angelegt und alles installiert.
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto check
echo Richte Bildwerk ein (einmalig, dauert ein bis zwei Minuten) ...
rem Bevorzugt eine Python-Version, fuer die es alle Pakete fertig gibt
for %%v in (3.13 3.14 3.12 3.11) do (
  if not exist ".venv\Scripts\python.exe" py -%%v -m venv .venv >nul 2>&1
)
if not exist ".venv\Scripts\python.exe" py -3 -m venv .venv 2>nul
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
echo.
echo Installiere die lokale KI (ONNX Runtime mit DirectML fuer die Grafikkarte) ...
".venv\Scripts\python.exe" -m pip install -e ".[local]"
if errorlevel 1 goto nolocal
type nul > ".venv\.installed"
goto run

:nolocal
type nul > ".venv\.installed"
echo.
echo Hinweis: Die lokale KI liess sich nicht installieren. Bildwerk startet trotzdem,
echo nur das KI-Schaerfen auf diesem PC fehlt. Abhilfe: Python 3.13 installieren,
echo den Ordner .venv loeschen und start.bat erneut starten.
echo.
pause

:run
".venv\Scripts\python.exe" -m bildwerk %*
if errorlevel 1 pause
exit /b

:nopython
echo.
echo Python wurde nicht gefunden.
echo Bitte Python 3.13 von https://www.python.org/downloads/ installieren
echo (dabei "Add python.exe to PATH" anhaken) und start.bat erneut starten.
pause
exit /b 1

:failed
echo.
echo Die Installation ist fehlgeschlagen. Bitte die Meldungen oben pruefen.
pause
exit /b 1
