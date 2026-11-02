VENV := $(CURDIR)/.venv
PYTHON := $(VENV)/bin/python
JAVA_HOME := $(CURDIR)/.tools/java
export JAVA_HOME
export PATH := $(JAVA_HOME)/bin:$(PATH)
export PYSPARK_PYTHON := $(PYTHON)
export PYSPARK_DRIVER_PYTHON := $(PYTHON)

.PHONY: setup setup-java install download prepare baseline test lint check clean-generated

setup:
	python3.11 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[data,dev]"
	$(PYTHON) scripts/setup_java.py

setup-java:
	$(PYTHON) scripts/setup_java.py

install:
	$(PYTHON) -m pip install -e ".[data,dev]"

download:
	$(PYTHON) -m recommender.data.download

prepare:
	$(PYTHON) -m recommender.data.prepare

baseline:
	$(PYTHON) -m recommender.baselines.popularity

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

check: lint test

clean-generated:
	@echo "Generated data and reports are intentionally not deleted automatically."
	@echo "Remove specific paths manually after confirming their contents."
