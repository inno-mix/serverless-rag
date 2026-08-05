# Serverless RAG — common tasks.
#
#   make check     the three gates CI runs
#   make deploy    preflight, then deploy everything
#   make destroy   remove everything, and verify it is gone

.PHONY: help install check lint test synth deploy preflight smoke destroy clean

PY := python3.12
VENV := .venv
BIN := $(VENV)/bin

help:
	@printf '\nServerless RAG\n\n'
	@printf '  make install    create .venv and install dependencies\n'
	@printf '  make check      lint, tests, and cdk synth (what CI runs)\n'
	@printf '  make preflight  check credentials, bootstrap, Bedrock access\n'
	@printf '  make deploy     preflight, then deploy all five stacks\n'
	@printf '  make smoke      end-to-end test against the deployed stack\n'
	@printf '  make destroy    delete everything, then verify nothing remains\n'
	@printf '  make clean      remove build artefacts (no AWS resources)\n\n'

install:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --quiet --upgrade pip
	$(BIN)/pip install --quiet -r requirements.txt
	@printf '\nReady. Activate with: source $(BIN)/activate\n'

check: lint test synth
	@printf '\nAll gates passed.\n'

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

test:
	$(BIN)/pytest tests/unit tests/infra tests/eval -q --cov=src/shared --cov-fail-under=80

synth:
	npx cdk synth --quiet

preflight:
	@./scripts/deploy.sh --check

deploy:
	@./scripts/deploy.sh

smoke:
	$(BIN)/pytest tests/integration -q

destroy:
	@./scripts/destroy.sh

clean:
	rm -rf cdk.out .pytest_cache .coverage cdk-outputs.json
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
