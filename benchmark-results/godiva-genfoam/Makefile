PYTHON ?= python
PYTHONPATH := ../..

.PHONY: all figures clean

all: godiva-genfoam.pdf

figures:
	MPLCONFIGDIR=/tmp/garnet-matplotlib PYTHONPATH=$(PYTHONPATH) $(PYTHON) make_figures.py

build/report.pdf: report.tex figures
	mkdir -p build
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build report.tex
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build report.tex

godiva-genfoam.pdf: build/report.pdf
	cp build/report.pdf $@

clean:
	rm -rf build godiva-genfoam.pdf figures tables
