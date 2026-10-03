.PHONY: install lint fmt type test check calibrate train
install:
	pip install -e ".[dev,doom]"
lint:
	ruff check . && ruff format --check .
fmt:
	ruff check --fix . && ruff format .
type:
	mypy
test:
	pytest
check: lint type test
train:
	doom-esn train --config configs/basic.yaml --out runs/basic
