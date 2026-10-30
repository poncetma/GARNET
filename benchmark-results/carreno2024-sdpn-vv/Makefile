PYTHON ?= python
PYTHONPATH := ../..

.PHONY: all figures clean

all: carreno2024-sdpn-vv.pdf

figures:
	MPLCONFIGDIR=/tmp/garnet-matplotlib PYTHONPATH=$(PYTHONPATH) $(PYTHON) make_figures.py

build/report.pdf: report.tex sections/*.tex figures
	mkdir -p build
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build report.tex
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build report.tex

carreno2024-sdpn-vv.pdf: build/report.pdf
	cp build/report.pdf $@

clean:
	rm -rf build carreno2024-sdpn-vv.pdf figures tables
