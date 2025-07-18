# setup
.PHONY: install
install: copy-templates
	uv sync --no-dev

.PHONY: install-dev
install-dev: copy-templates
	uv sync
	pre-commit install

.PHONY: copy-templates
copy-templates:
	@if [ ! -f spd/user_metrics_and_figs.py ]; then \
		cp spd/user_metrics_and_figs.py.example spd/user_metrics_and_figs.py; \
		echo "Created spd/user_metrics_and_figs.py from template"; \
	fi
	@if [ ! -f spd/scripts/sweep_params.yaml ]; then \
		cp spd/scripts/sweep_params.yaml.example spd/scripts/sweep_params.yaml; \
		echo "Created spd/scripts/sweep_params.yaml from template"; \
	fi


# checks
.PHONY: type
type: temp-user-metrics
	basedpyright

.PHONY: format
format: temp-user-metrics
	# Fix all autofixable problems (which sorts imports) then format errors
	ruff check --fix
	ruff format

.PHONY: check
check: format type

.PHONY: check-pre-commit
check-pre-commit:
	SKIP=no-commit-to-branch pre-commit run -a --hook-stage commit

# tests
.PHONY: test
test: temp-user-metrics
	pytest tests/

.PHONY: test-all
test-all: temp-user-metrics
	pytest tests/ --runslow

COVERAGE_DIR=docs/coverage

.PHONY: coverage
coverage:
	uv run pytest tests/ --cov=spd --runslow
	mkdir -p $(COVERAGE_DIR)
	uv run python -m coverage report -m > $(COVERAGE_DIR)/coverage.txt
	uv run python -m coverage html --directory=$(COVERAGE_DIR)/html/

# making sure `spd/user_metrics_and_figs.py.template` is checked
# by copying it to a temporary directory
TEMP_DIR = tests/.temp

.PHONY: temp-user-metrics-clean
temp-user-metrics-clean:
	rm -rf $(TEMP_DIR)

.PHONY: temp-user-metrics-copy
temp-user-metrics-copy: temp-user-metrics-clean
	mkdir -p $(TEMP_DIR)
	cp spd/user_metrics_and_figs.py.example $(TEMP_DIR)/user_metrics_and_figs.py
	echo "Created temporary $(TEMP_DIR)/user_metrics_and_figs.py from template for testing purposes"
