@echo off
REM Windows batch file for running Ring Attractor tests

setlocal enabledelayedexpansion

echo Ring Attractor Test Runner (Windows)
echo ====================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found in PATH
    echo Please install Python and add it to your PATH
    pause
    exit /b 1
)

REM Parse command line arguments
set "COMMAND=all"
set "COVERAGE="
set "VERBOSE="
set "FAST="
set "PARALLEL="
set "HTML_REPORT="

:parse
if "%~1"=="" goto endparse
if "%~1"=="unit" set "COMMAND=unit"
if "%~1"=="integration" set "COMMAND=integration"
if "%~1"=="all" set "COMMAND=all"
if "%~1"=="coverage" set "COVERAGE=--coverage"
if "%~1"=="verbose" set "VERBOSE=--verbose"
if "%~1"=="fast" set "FAST=--fast"
if "%~1"=="parallel" set "PARALLEL=--parallel 4"
if "%~1"=="html" set "HTML_REPORT=--html-report"
if "%~1"=="help" goto help
shift
goto parse

:endparse

REM Check if pytest is installed
python -c "import pytest" >nul 2>&1
if errorlevel 1 (
    echo Warning: pytest not found
    echo Installing test dependencies...
    pip install -r requirements-test.txt
    if errorlevel 1 (
        echo Error: Failed to install test dependencies
        pause
        exit /b 1
    )
)

REM Build the test command
set "TEST_CMD=python run_tests.py"

if "%COMMAND%"=="unit" (
    set "TEST_CMD=!TEST_CMD! --unit"
) else if "%COMMAND%"=="integration" (
    set "TEST_CMD=!TEST_CMD! --integration"
) else (
    set "TEST_CMD=!TEST_CMD! --all"
)

if defined COVERAGE set "TEST_CMD=!TEST_CMD! !COVERAGE!"
if defined VERBOSE set "TEST_CMD=!TEST_CMD! !VERBOSE!"
if defined FAST set "TEST_CMD=!TEST_CMD! !FAST!"
if defined PARALLEL set "TEST_CMD=!TEST_CMD! !PARALLEL!"
if defined HTML_REPORT set "TEST_CMD=!TEST_CMD! !HTML_REPORT!"

echo Running command: !TEST_CMD!
echo.

REM Execute the test command
!TEST_CMD!

set "EXIT_CODE=!errorlevel!"

echo.
if !EXIT_CODE! equ 0 (
    echo [SUCCESS] Tests completed successfully
    if defined COVERAGE (
        echo.
        echo Coverage report available at: test_results\htmlcov\index.html
    )
) else (
    echo [FAILED] Tests failed
)

pause
exit /b !EXIT_CODE!

:help
echo.
echo Usage: test.bat [options]
echo.
echo Options:
echo   unit          Run unit tests only
echo   integration   Run integration tests only  
echo   all           Run all tests (default)
echo   coverage      Generate coverage report
echo   verbose       Verbose output
echo   fast          Skip slow tests
echo   parallel      Run tests in parallel
echo   html          Generate HTML coverage report
echo   help          Show this help
echo.
echo Examples:
echo   test.bat                    Run all tests
echo   test.bat unit               Run unit tests only
echo   test.bat coverage html      Run with coverage and HTML report
echo   test.bat fast parallel      Run fast tests in parallel
echo.
pause
exit /b 0