@echo off
REM Load .env from the project root if it exists.
REM Requires delayed expansion for substring checks on FOR loop variables.
if not exist "%PIXI_PROJECT_ROOT%\.env" goto :eof

setlocal enabledelayedexpansion
for /f "usebackq tokens=1,* delims==" %%A in ("%PIXI_PROJECT_ROOT%\.env") do (
    set "_line=%%A"
    if not "!_line!"=="" if not "!_line:~0,1!"=="#" set "%%A=%%B"
)
endlocal
:eof
