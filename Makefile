.PHONY: help install docker-up docker-down docker-logs lint format test clean mlflow-sync mlflow-ui

help:
	@echo "PromoSense Development Commands:"
	@echo "  make install      Install Python dependencies in current environment"
	@echo "  make docker-up    Start PostgreSQL and MLflow containers in background"
	@echo "  make docker-down  Stop all Docker containers"
	@echo "  make docker-logs  Tail logs from Docker containers"
	@echo "  make mlflow-sync  Synchronize all frozen models and benchmarks into MLflow"
	@echo "  make mlflow-ui    Start local MLflow UI server"
	@echo "  make lint         Run Ruff linter and static analysis"
	@echo "  make format       Auto-format code and sort imports"
	@echo "  make test         Run pytest suite across ML pipeline and backend"
	@echo "  make clean        Remove cache and coverage artifacts"

install:
	uv pip install -e ".[dev]" || pip install -e ".[dev]"

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

mlflow-sync:
	python -m ml_pipeline.src.models.mlflow_tracker

mlflow-ui:
	mlflow ui --port 5000

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache', '.mypy_cache', 'htmlcov']]"
