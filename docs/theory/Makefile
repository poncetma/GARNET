# Build the GARNET theory report from data/*.json (written by run_cases.py).
PY ?= /opt/psi/conda-envs/x86_64/ra-standard_py312/bin/python

garnet-theory.pdf: report.tex make_figures.py $(wildcard data/*.json)
	$(PY) make_figures.py
	mkdir -p build
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build report.tex >/dev/null
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build report.tex >/dev/null
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build report.tex | grep -iE "warning|overfull" || true
	cp build/report.pdf $@

clean:
	rm -rf build figures tables

.PHONY: clean
