PYTHON ?= .venv/bin/python
LATEX ?= pdflatex
BIBTEX ?= bibtex

.PHONY: figures paper test

figures:
	$(PYTHON) figures.py results paper/figures

paper: figures
	mkdir -p out/paper-build output/pdf
	cd paper && $(LATEX) -interaction=nonstopmode -halt-on-error -output-directory=../out/paper-build manuscript.tex
	cd out/paper-build && BIBINPUTS="$(CURDIR)/paper:" $(BIBTEX) manuscript
	cd paper && $(LATEX) -interaction=nonstopmode -halt-on-error -output-directory=../out/paper-build manuscript.tex
	cd paper && $(LATEX) -interaction=nonstopmode -halt-on-error -output-directory=../out/paper-build manuscript.tex
	cp out/paper-build/manuscript.pdf output/pdf/habs_arousal.pdf

test:
	$(PYTHON) -m pytest -q
