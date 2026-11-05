VENV := $(CURDIR)/.venv
PYTHON := $(VENV)/bin/python
JAVA_HOME := $(CURDIR)/.tools/java
export JAVA_HOME
export PATH := $(JAVA_HOME)/bin:$(PATH)
export PYSPARK_PYTHON := $(PYTHON)
export PYSPARK_DRIVER_PYTHON := $(PYTHON)
export TF_USE_LEGACY_KERAS := 1

.PHONY: setup setup-java install install-retrieval install-indexing install-serving download prepare features baseline prepare-retrieval train-retrieval build-index prepare-ranking train-ranking test lint check clean-generated

setup:
	python3.11 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[data,dev]"
	$(PYTHON) scripts/setup_java.py

setup-java:
	$(PYTHON) scripts/setup_java.py

install:
	$(PYTHON) -m pip install -e ".[data,dev]"

install-retrieval:
	$(PYTHON) -m pip install -e ".[retrieval]"

install-indexing:
	$(PYTHON) -m pip install -e ".[indexing]"

install-serving:
	$(PYTHON) -m pip install -e ".[serving]"

download:
	$(PYTHON) -m recommender.data.download

prepare:
	$(PYTHON) -m recommender.data.prepare

features:
	$(PYTHON) -m recommender.features.build

baseline:
	$(PYTHON) -m recommender.baselines.popularity

prepare-retrieval:
	$(PYTHON) -m recommender.retrieval.prepare

train-retrieval:
	$(PYTHON) -m recommender.retrieval.train

build-index:
	$(PYTHON) -m recommender.indexing.build

prepare-ranking:
	$(PYTHON) -m recommender.ranking.prepare

train-ranking:
	$(PYTHON) -m recommender.ranking.train

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

check: lint test

clean-generated:
	@echo "Generated data and reports are intentionally not deleted automatically."
	@echo "Remove specific paths manually after confirming their contents."
