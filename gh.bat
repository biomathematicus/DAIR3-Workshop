@echo off
setlocal enabledelayedexpansion

rem ================================================================
rem  gh.bat  --  Git helper with automatic version stamping
rem
rem  Usage:
rem    gh                    Pull only; print session verification prompt
rem    gh "commit message"   Stage, pull, commit, push, stamp, re-push
rem
rem  The GITCOMMIT hash printed in the prompt is the one written into
rem  poecilia_manuscript.tex and VERSION -- not the stamp commit hash.
rem  Claude searches for that string in the project knowledge base.
rem ================================================================

rem set TEXFILE=GITCOMMIT.txt
set VERSIONFILE=VERSION.txt

rem ================================================================
rem  PULL-ONLY MODE  (no arguments)
rem ================================================================
if "%~1"=="" (
    echo [gh] No commit message supplied -- pulling only.
    git pull origin main
    for /f "tokens=1" %%i in (!VERSIONFILE!) do set WORKHASH=%%i
    echo [gh] Stamped commit in files: !WORKHASH!
    call :PRINT_PROMPT
    goto :EOF
)

rem ================================================================
rem  COMMIT MODE
rem ================================================================

rem ---- Stage all changes ----
git add -A

rem ---- Pull before committing to avoid divergence ----
git pull origin main

rem ---- Commit only if there is something staged ----
git diff --cached --quiet
if errorlevel 1 (
    git commit -m %*
) else (
    echo [gh] Nothing to commit -- working tree clean.
)

rem ---- Push to DAIR3 organization repo ----
git push origin HEAD

rem ---- Push to biomathematicus repo ----
git push biomathematicus HEAD

rem ---- Capture the WORK hash (this is what goes into the files) ----
for /f %%i in ('git rev-parse --short HEAD') do set WORKHASH=%%i
echo [gh] Work commit: !WORKHASH!

rem ================================================================
rem  VERSION STAMP
rem ================================================================

rem ---- Write VERSION ----
echo !WORKHASH! %DATE% %TIME%> !VERSIONFILE!

rem ---- Update % GITCOMMIT: line in the .tex file ----
rem if exist !TEXFILE! (
rem     powershell -NoProfile -Command "(Get-Content '!TEXFILE!') -replace 'GITCOMMIT:.*', 'GITCOMMIT: !WORKHASH!' | Set-Content '!TEXFILE!'"
rem     echo [gh] Updated GITCOMMIT in !TEXFILE!
rem ) else (
rem     echo [gh] WARNING: !TEXFILE! not found -- GITCOMMIT line not updated.
rem )

rem ---- Stage and commit stamp files ----
rem git add !VERSIONFILE! !TEXFILE!
git add !VERSIONFILE! 
git diff --cached --quiet
if errorlevel 1 (
    git commit -m "Auto: version stamp !WORKHASH!"
    rem ---- Push to DAIR3 organization repo ----
    git push origin HEAD

    rem ---- Push to biomathematicus repo ----
    git push biomathematicus HEAD
    echo [gh] Stamp commit pushed.
)

rem ---- Print the agent session prompt (always uses WORKHASH) ----
call :PRINT_PROMPT
goto :EOF

rem ================================================================
rem  SUBROUTINE: print copy-paste agent verification prompt
rem ================================================================
:PRINT_PROMPT
echo.
echo =================================================================
echo          CONFIGURATION MANAGEMENT ACROSS BOUNDARIES
echo =================================================================
echo.
echo   Before evaluating any inputs or producing outputs, search
echo   project knowledge for "GITCOMMIT !WORKHASH!" and confirm
echo   that the search returns a result containing that exact string.
echo.
echo   Current commit: !WORKHASH!
echo =================================================================
echo.
echo   If the search returns an older hash or nothing, the project
echo   knowledge base has not yet re-indexed -- wait ~5 minutes and
echo   run  gh  (pull-only) to get a fresh prompt before proceeding.
echo.

goto :EOF