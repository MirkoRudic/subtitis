@echo off
rem ==========================================================================
rem  Subtitis — dvoklik i radi.
rem
rem  Diže dashboard i obradu u ISTOM procesu i otvara pregledač. Nema dva
rem  prozora, nema `subtitis worker` sa strane.
rem
rem  Prozor koji se otvori JESTE server: dok stoji, Subtitis radi. Zatvaranje
rem  prozora (ili Ctrl+C) gasi i obradu.
rem ==========================================================================
setlocal

cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo.
    echo   Nema virtuelnog okruzenja: %PY%
    echo.
    echo   Napravi ga jednom, iz ovog foldera:
    echo       python -m venv .venv
    echo       .venv\Scripts\python -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

"%PY%" -m subtitis serve --open %*
set "RC=%ERRORLEVEL%"

rem Prozor ostaje otvoren kad server padne, da se poruka o gresci procita.
if not "%RC%"=="0" (
    echo.
    echo   Server je stao, izlazni kod %RC%.
    pause
)
exit /b %RC%
