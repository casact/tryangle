.PHONY: venv setup ruff format test check clean build test-publish

SHELL := /bin/bash

PYTHON_INTERPRETER := python
VENV := source .venv/bin/activate
PROJECT_CONFIG := pyproject.toml

venv: .venv/touchfile

.venv/touchfile: requirements.txt requirements-dev.txt
	$(VENV); pip-sync requirements.txt requirements-dev.txt
	$(VENV); pip install -e .
	touch .venv/touchfile

requirements.txt: $(PROJECT_CONFIG)
	$(VENV); pip-compile --output-file=requirements.txt --resolver=backtracking $(PROJECT_CONFIG)

requirements-dev.txt: $(PROJECT_CONFIG)
	$(VENV); pip-compile --extra=dev --output-file=requirements-dev.txt --resolver=backtracking $(PROJECT_CONFIG)

test: venv
	$(VENV); tox

build:
	rm -rf build dist
	$(VENV); python -m build

test-publish:
	$(VENV); python -m twine upload --repository testpypi dist/* --verbose

publish:
	$(VENV); python -m twine --repository tryangle dist/* --verbose

setup:
	virtualenv .venv
	$(VENV); pip install --upgrade pip setuptools wheel build
	$(VENV); pip install pip-tools

ruff:
	$(VENV); ruff .

format:
	$(VENV); ruff .
	$(VENV); black .

check:
	$(VENV); ruff check .
	$(VENV); black --check .

clean:
	find . -type d -name ".ipynb_checkpoints" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf build dist