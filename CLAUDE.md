# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
SPD (Stochastic Parameter Decomposition) is a research framework for analyzing neural network components and their interactions through sparse parameter decomposition techniques. The codebase supports three experimental domains: TMS (Toy Model of Superposition), ResidMLP (residual MLP analysis), and Language Models.

## Development Commands

**Setup:**
- `make install-dev` - Install package with dev dependencies and pre-commit hooks
- `make install` - Install package only (`pip install -e .`)

Both installation commands automatically create `spd/user_metrics_and_figs.py` from `spd/user_metrics_and_figs.py.example` if it doesn't exist.

**Code Quality:**
- `make check` - Run full pre-commit suite (pyright, ruff lint, ruff format)
- `make type` - Run pyright type checking only
- `make format` - Run ruff linter and formatter

**Testing:**
- `make test` - Run tests (excluding slow tests)
- `make test-all` - Run all tests including slow ones
- `python -m pytest tests/test_specific.py` - Run specific test file
- `python -m pytest tests/test_specific.py::test_function` - Run specific test

## Architecture Overview

**Core SPD Framework:**
- `spd/run_spd.py` - Main SPD optimization logic called by all experiments
- `spd/configs.py` - Pydantic config classes for all experiment types
- `spd/registry.py` - Centralized experiment registry with all experiment configurations
- `spd/models/component_model.py` - Core ComponentModel that wraps target models
- `spd/models/components.py` - Component types (LinearComponent, EmbeddingComponent, etc.)
- `spd/losses.py` - SPD loss functions (faithfulness, reconstruction, importance minimality)
- `spd/user_metrics_and_figs.py` - User-defined metrics and visualizations (created from template)

**Experiment Structure:**
Each experiment (`spd/experiments/{tms,resid_mlp,lm}/`) contains:
- `models.py` - Experiment-specific model classes and pretrained loading
- `*_decomposition.py` - Main SPD execution script
- `train_*.py` - Training script for target models  
- `*_config.yaml` - Configuration files
- `plotting.py` - Visualization utilities

**Key Data Flow:**
1. Experiments load pretrained target models via WandB or local paths
2. Target models are wrapped in ComponentModel with specified target modules
3. SPD optimization runs via `spd.run_spd.optimize()` with config-driven loss combination
4. Results include component masks, causal importance scores, and visualizations

**Configuration System:**
- YAML configs define all experiment parameters
- Pydantic models provide type safety and validation  
- WandB integration for experiment tracking and model storage
- Supports both local paths and `wandb:project/runs/run_id` format for model loading
- Centralized experiment registry (`spd/registry.py`) manages all experiment configurations

**Component Analysis:**
- Components represent sparse decompositions of target model parameters
- Stochastic masking enables differentiable sparsity
- Causal importance quantifies component contributions to model outputs
- Multiple loss terms balance faithfulness, reconstruction quality, and sparsity

**Environment setup:**
- Requires `.env` file with WandB credentials (see `.env.example`)
- Uses WandB for experiment tracking and model storage
- All runs generate timestamped output directories with configs, models, and plots

## Common Usage Patterns

**Running SPD experiments:**
```bash
uv run spd/experiments/tms/tms_decomposition.py spd/experiments/tms/tms_5-2_config.yaml
uv run spd/experiments/resid_mlp/resid_mlp_decomposition.py spd/experiments/resid_mlp/resid_mlp1_config.yaml
uv run spd/experiments/lm/lm_decomposition.py spd/experiments/lm/ss_emb_config.yaml
```

A run will output the important losses and the paths to which important figures are saved. Use these
to analyse the result of the runs.

**Custom Metrics and Visualizations:**
The `spd/user_metrics_and_figs.py` file (automatically created from template during installation) allows adding custom metrics and visualizations without modifying core framework code. The file provides:
- `compute_user_metrics()` - Define metrics logged to WandB during optimization
- `create_user_figures()` - Create matplotlib figures saved during optimization

Both functions receive the component model, gates, causal importances, and other optimization state, allowing flexible analysis of SPD results.

**Sweeps**
Run hyperparameter sweeps using WandB on the GPU cluster:

```bash
spd-sweep <experiment_name> <n_agents> [--cpu] [--job_suffix <suffix>]
```

Examples:
```bash
spd-sweep tms_5-2 5                    # Run TMS 5-2 sweep with 5 GPU agents
spd-sweep resid_mlp1 3 --cpu           # Run ResidMLP1 sweep with 3 CPU agents  
spd-sweep tms_5-2-id 2 --job_suffix test   # Run with custom job suffix
```

**Supported experiments:** tms_5-2, tms_5-2-id, tms_40-10, tms_40-10-id, resid_mlp1, resid_mlp2, resid_mlp3

**How it works:**
1. Creates a WandB sweep using parameters from `spd/sweeps/sweep_params.yaml`
2. Deploys multiple SLURM agents as a job array to run the sweep
3. Each agent runs on a single GPU by default (use `--cpu` for CPU-only)
4. Creates a git snapshot to ensure consistent code across all agents

**Before running:** Update `spd/sweeps/sweep_params.yaml` with desired sweep parameters.

**Logs:** Agent logs are found in `~/slurm_logs/slurm-<job_id>_<task_id>.out`

**Evaluations**
Run systematic evaluations across multiple experiments:

```bash
spd-evals                                                  # Run all experiments
spd-evals --experiments tms_5-2,resid_mlp3,ss_emb          # Run specific experiments
spd-evals tms_5-2,resid_mlp1 --job-suffix my_suffix        # Add job suffix
```

**How it works:**
1. Runs selected experiments as a SLURM job array with unique `evals_id`
2. Creates a WandB report with visualizations for each experiment type
3. By default, runs analysis job after all evaluations complete
4. Uses git snapshot for consistent code across all experiments

**Features:**
- Automatic WandB report generation with causal importance plots and loss curves
- Analysis job dependency (runs after all evaluations complete)
- Comprehensive logging and job tracking
- Support for experiment filtering

**Logs:** 
- Array job logs: `~/slurm_logs/slurm-<array_job_id>_<task_id>.out`
- Analysis job log: `~/slurm_logs/slurm-<analysis_job_id>.out`

**Cluster Usage Guidelines:**
- DO NOT use more than 8 GPUs at one time
- This includes not setting off multiple sweeps/evals that total >8 GPUs
- Monitor jobs with: `squeue --format="%.18i %.9P %.15j %.12u %.12T %.10M %.9l %.6D %b %R" --me`

## Coding Guidelines
- Always use the PEP 604 typing format of unions everywhere and no "Optional".