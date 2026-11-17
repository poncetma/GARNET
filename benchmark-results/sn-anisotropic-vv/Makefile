# Build: `module load texlive/2015 && make` (the system TeX lacks booktabs,
# geometry and other packages pandoc's LaTeX template needs).
PYTHON ?= python
PYTHONPATH := ../..

.PHONY: all figures clean

all: sn-anisotropic-vv.pdf

figures:
	MPLCONFIGDIR=/tmp/garnet-matplotlib PYTHONPATH=$(PYTHONPATH) $(PYTHON) make_figures.py

sn-anisotropic-vv.pdf: report.md figures/convergence.pdf
	pandoc report.md -o $@ --pdf-engine=pdflatex --resource-path=.

clean:
	rm -f sn-anisotropic-vv.pdf
