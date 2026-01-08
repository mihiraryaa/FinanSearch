#!/bin/bash

echo "Compiling LaTeX paper..."
echo

pdflatex main.tex
if [ $? -ne 0 ]; then
    echo "First compilation failed!"
    exit 1
fi

echo
echo "Running second pass for references..."
pdflatex main.tex
if [ $? -ne 0 ]; then
    echo "Second compilation failed!"
    exit 1
fi

echo
echo "Compilation successful! PDF created: main.pdf"
echo

# Clean up auxiliary files
rm -f *.aux *.log *.out

echo "Cleaned up auxiliary files"
