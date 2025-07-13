RUN = uv run

.PHONY: install
install:
	uv sync --no-dev
	@if [ ! -f spd/user_metrics_and_figs.py ]; then \
		cp spd/user_metrics_and_figs.py.example spd/user_metrics_and_figs.py; \
		echo "Created spd/user_metrics_and_figs.py from template"; \
	fi
	@if [ ! -f spd/scripts/sweep_params.yaml ]; then \
		cp spd/scripts/sweep_params.yaml.example spd/scripts/sweep_params.yaml; \
		echo "Created spd/scripts/sweep_params.yaml from template"; \
	fi

.PHONY: install-dev
install-dev:
	uv sync
	$(RUN) pre-commit install
	@if [ ! -f spd/user_metrics_and_figs.py ]; then \
		cp spd/user_metrics_and_figs.py.example spd/user_metrics_and_figs.py; \
		echo "Created spd/user_metrics_and_figs.py from template"; \
	fi
	@if [ ! -f spd/scripts/sweep_params.yaml ]; then \
		cp spd/scripts/sweep_params.yaml.example spd/scripts/sweep_params.yaml; \
		echo "Created spd/scripts/sweep_params.yaml from template"; \
	fi

.PHONY: type
type:
	SKIP=no-commit-to-branch $(RUN) pre-commit run -a basedpyright

.PHONY: format
format:
	# Fix all autofixable problems (which sorts imports) then format errors
	SKIP=no-commit-to-branch $(RUN) pre-commit run -a ruff-lint
	SKIP=no-commit-to-branch $(RUN) pre-commit run -a ruff-format

.PHONY: check
check:
	SKIP=no-commit-to-branch $(RUN) pre-commit run -a --hook-stage commit

.PHONY: test
test:
	$(RUN) pytest tests/

.PHONY: test-all
test-all:
	$(RUN) pytest tests/ --runslow