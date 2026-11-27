TAO_CORE ?= ../tao-core

.PHONY: dev test wheels
dev:
	uv venv --allow-existing .venv
	uv pip install --python .venv -e $(TAO_CORE)/khai -e '.[dev]'

test:
	.venv/bin/python -m pytest

wheels:
	uv build --wheel --out-dir dist .
	uv build --wheel --out-dir dist $(TAO_CORE)/khai
