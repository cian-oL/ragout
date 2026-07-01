.PHONY: setup run-service-api

setup:
	uv sync
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg
	uv run pre-commit run --all-files || true

run-service-api:
	uv run --package service service-api
