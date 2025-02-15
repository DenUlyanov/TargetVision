setup:
	uv venv .venv && source .venv/bin/activate && uv pip install --editable .

install:
	@echo "Installing dependencies..."
	uv pip install -r requirements.txt

update-deps:
	@echo "Updating dependencies and regenerating requirements.txt..."
	uv pip compile pyproject.toml--output-file=requirements.txt
	uv pip install -r requirements.txt


test:
	@echo "Running tests..."
	PYTHONPATH=$(pwd)/src .venv/bin/python -m pytest --import-mode=importlib --tb=short --verbose

format:
	@echo "Formatting code with Black..."
	uv run black .

type-check:
	@echo "Type checking code with Mypy..."
	uv run mypy .

lint:
	@echo "Linting code with Ruff..."
	uv run ruff check --fix .

check: format lint test
