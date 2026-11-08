VENV := $(CURDIR)/.venv
PYTHON := $(VENV)/bin/python
JAVA_HOME := $(CURDIR)/.tools/java
CONFIG ?= configs/dev.toml
export JAVA_HOME
export PATH := $(JAVA_HOME)/bin:$(PATH)
export PYSPARK_PYTHON := $(PYTHON)
export PYSPARK_DRIVER_PYTHON := $(PYTHON)
export TF_USE_LEGACY_KERAS := 1
export RECSYS_CONFIG := $(CONFIG)

.PHONY: setup setup-java install lock install-retrieval install-indexing install-serving download prepare features baseline prepare-retrieval train-retrieval build-index prepare-ranking train-ranking export-triton verify-triton test-triton infra-up infra-down demo-up demo-down test-redis validate-pipeline airflow-test serve benchmark-serving test test-unit test-integration test-coverage lint check check-fast clean-generated

setup:
	python3.11 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install -e ".[all,dev]" --no-deps
	$(PYTHON) scripts/setup_java.py

setup-java:
	$(PYTHON) scripts/setup_java.py

install:
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install -e ".[all,dev]" --no-deps

lock:
	$(VENV)/bin/pip-compile --all-extras --strip-extras --resolver=backtracking --output-file=requirements.lock pyproject.toml
	$(VENV)/bin/pip-compile --extra=retrieval --extra=indexing --extra=serving --strip-extras --resolver=backtracking --output-file=requirements-api.lock pyproject.toml

install-retrieval:
	$(PYTHON) -m pip install -e ".[retrieval]" -c requirements.lock

install-indexing:
	$(PYTHON) -m pip install -e ".[indexing]" -c requirements.lock

install-serving:
	$(PYTHON) -m pip install -e ".[serving]" -c requirements.lock

download:
	$(PYTHON) -m recommender.data.download

prepare:
	$(PYTHON) -m recommender.data.prepare --config $(CONFIG)

features:
	$(PYTHON) -m recommender.features.build --config $(CONFIG)

baseline:
	$(PYTHON) -m recommender.baselines.popularity --config $(CONFIG)

prepare-retrieval:
	$(PYTHON) -m recommender.retrieval.prepare --config $(CONFIG)

train-retrieval:
	$(PYTHON) -m recommender.retrieval.train --config $(CONFIG)

build-index:
	$(PYTHON) -m recommender.indexing.build --config $(CONFIG)

prepare-ranking:
	$(PYTHON) -m recommender.ranking.prepare --config $(CONFIG)

train-ranking:
	$(PYTHON) -m recommender.ranking.train --config $(CONFIG)

export-triton:
	$(PYTHON) -m recommender.serving.triton_repository

verify-triton:
	$(PYTHON) -m recommender.serving.verify_triton

test-triton:
	RUN_TRITON_INTEGRATION=1 $(PYTHON) -m pytest tests/integration/test_triton.py -v

infra-up: export-triton
	docker compose up -d redis triton

infra-down:
	docker compose down

demo-up:
	docker compose --profile api up --build -d redis api

demo-down:
	docker compose --profile api down

test-redis:
	RUN_REDIS_INTEGRATION=1 $(PYTHON) -m pytest tests/integration/test_redis.py -v

validate-pipeline:
	@for stage in validate_raw prepare_data build_features prepare_retrieval train_retrieval build_index prepare_ranking train_ranking export_triton validate_serving; do \
		$(PYTHON) scripts/pipeline_stage.py --stage $$stage --mode validate || exit 1; \
	done

airflow-test:
	mkdir -p .airflow
	docker compose --profile orchestration run --rm airflow airflow db migrate
	docker compose --profile orchestration run --rm airflow airflow dags test recommender_retraining 2026-11-05

serve:
	$(PYTHON) -m recommender.serving.api

benchmark-serving:
	$(PYTHON) -m recommender.serving.benchmark

test:
	$(PYTHON) -m pytest

test-unit:
	$(PYTHON) -m pytest tests/unit

test-integration:
	$(PYTHON) -m pytest tests/integration

test-coverage:
	$(PYTHON) -m pytest tests/unit --cov --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check .

check: lint test

check-fast: lint test-unit

clean-generated:
	@echo "Generated data and reports are intentionally not deleted automatically."
	@echo "Remove specific paths manually after confirming their contents."
