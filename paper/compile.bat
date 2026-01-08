@echo off
echo Compiling LaTeX paper...
echo.

pdflatex main.tex
if %errorlevel% neq 0 (
    echo First compilation failed!
    pause
    exit /b 1
)

echo.
echo Running second pass for references...
pdflatex main.tex
if %errorlevel% neq 0 (
    echo Second compilation failed!
    pause
    exit /b 1
)

echo.
echo Compilation successful! PDF created: main.pdf
echo.
pause
