RUN = uv run

.PHONY: install
install: copy-templates
	uv sync --no-dev

.PHONY: install-dev
install-dev: copy-templates
	uv sync
	$(RUN) pre-commit install

# If the file "spd/user_metrics_and_figs.py" does not exist,
# copy it from the example file and print a confirmation message.
spd/user_metrics_and_figs.py:
	@cp spd/user_metrics_and_figs.py.example spd/user_metrics_and_figs.py
	@echo "Created spd/user_metrics_and_figs.py from template"

# Same logic as above: create the sweep_params.yaml file from its template
# only if it doesn't already exist.
spd/scripts/sweep_params.yaml:
	@cp spd/scripts/sweep_params.yaml.example spd/scripts/sweep_params.yaml
	@echo "Created spd/scripts/sweep_params.yaml from template"

# Declare "copy-templates" as a phony target (not a file on disk)
# This rule depends on two file targets. If either file doesn't exist,
# its associated recipe will be executed to create it.
.PHONY: copy-templates
copy-templates: spd/user_metrics_and_figs.py spd/scripts/sweep_params.yaml

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