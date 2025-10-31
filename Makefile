PY=python
PIP=pip

.PHONY: install install-dev lint type test format run pre-commit

install:
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -r requirements-dev.txt
	pre-commit install

lint:
	ruff check .
	black --check .

type:
	mypy .

test:
	pytest -q

format:
	black .
	ruff check --fix .

run:
	uvicorn api.main:app --reload