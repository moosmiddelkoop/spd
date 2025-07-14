# CLAUDE_local.md

Local environment notes for Claude Code (not tracked by git).

## Environment Setup

**IMPORTANT**: Always activate the virtual environment before running Python commands or tests:

```bash
source .venv/bin/activate
```

This is required for:
- Running Python scripts
- Running make commands (type checking, linting, tests)
- Installing dependencies
- Any git operations that trigger pre-commit hooks

## Common Commands After Activation

```bash
# Type checking
make type

# Run all checks (type, lint, format)
make check

# Run tests
make test

# Run spd-run locally
spd-run --experiments tms_5-2 --local
```

## ebatch Function Updates

The `ebatch` function in ~/.bashrc has been modified to automatically create a permissive slconf file if one doesn't exist. When no slconf is found, it creates one with:

- Current virtual environment (or defaults to `.venv`)
- 24-hour time limit
- 1 node
- 1 GPU
- `all` partition
- Output to ~/slurm_logs/

This allows you to quickly submit jobs without manually creating slconf files:
```bash
ebatch myjob "python script.py"
```