VENV := $(CURDIR)/.venv
PYTHON := $(VENV)/bin/python
JAVA_HOME := $(CURDIR)/.tools/java
export JAVA_HOME
export PATH := $(JAVA_HOME)/bin:$(PATH)
export PYSPARK_PYTHON := $(PYTHON)
export PYSPARK_DRIVER_PYTHON := $(PYTHON)
export TF_USE_LEGACY_KERAS := 1

.PHONY: setup setup-java install install-retrieval install-indexing install-serving download prepare features baseline prepare-retrieval train-retrieval build-index prepare-ranking train-ranking export-triton verify-triton test-triton infra-up infra-down test-redis validate-pipeline airflow-test serve benchmark-serving test lint check clean-generated

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

lint:
	$(PYTHON) -m ruff check .

check: lint test

clean-generated:
	@echo "Generated data and reports are intentionally not deleted automatically."
	@echo "Remove specific paths manually after confirming their contents."
