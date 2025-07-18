# SPD - Stochastic Parameter Decomposition
Code used in the paper [Stochastic Parameter Decomposition](https://arxiv.org/abs/2506.20790)

Weights and Bias [report](https://wandb.ai/goodfire/spd-tms/reports/SPD-paper-report--VmlldzoxMzE3NzU0MQ) accompanying the paper.

## Installation
From the root of the repository, run one of

```bash
make install-dev  # To install the package, dev requirements, pre-commit hooks, and create user files
make install  # To install the package (runs `pip install -e .`) and create user files
```

Both installation commands will automatically create `spd/user_metrics_and_figs.py` from `spd/user_metrics_and_figs.py.example` if it doesn't already exist. This file allows you to define custom metrics and visualizations for SPD experiments without modifying the core framework code.

## Usage
Place your wandb information in a .env file. See .env.example for an example.

The repository consists of several `experiments`, each of which containing scripts to run SPD,
analyse results, and optionally a train a target model:
- `spd/experiments/tms` - Toy model of superposition
- `spd/experiments/resid_mlp` - Toy model of compressed computation and toy model of distributed
  representations
- `spd/experiments/lm` - Language model loaded from huggingface.

Note that the `lm` experiment allows for running SPD on any model pulled from huggingface, provided
you only need to decompose `nn.Linear` or `nn.Embedding` layers (other layer types are not yet
supported, though these should cover most parameters).

### Run SPD

The unified `spd-run` command provides a single entry point for running SPD experiments, supporting
fixed configurations, parameter sweeps, and evaluation runs. All runs are tracked in W&B with
workspace views created for each experiment.

#### Individual Experiments
SPD can either be run by executing any of the `*_decomposition.py` scripts defined in the experiment
subdirectories, along with a corresponding config file. E.g.
```bash
uv run spd/experiments/tms/tms_decomposition.py spd/experiments/tms/tms_5-2_config.yaml
```

Or by using the unified runner:
```bash
spd-run --experiments tms_5-2                    # Run a specific experiment
spd-run --experiments tms_5-2,resid_mlp1         # Run multiple experiments
spd-run                                          # Run all experiments
```

**Available experiments** (defined in `spd/registry.py`):
- `tms_5-2` - TMS with 5 features, 2 hidden dimensions
- `tms_5-2-id` - TMS with 5 features, 2 hidden dimensions (fixed identity in-between)
- `tms_40-10` - TMS with 40 features, 10 hidden dimensions  
- `tms_40-10-id` - TMS with 40 features, 10 hidden dimensions (fixed identity in-between)
- `resid_mlp1` - ResidMLP with 1 layer
- `resid_mlp2` - ResidMLP with 2 layers
- `resid_mlp3` - ResidMLP with 3 layers

#### Sweeps
For running parameter sweeps on a SLURM cluster:

```bash
spd-run --experiments <experiment_name> --sweep --n_agents <n_agents> [--cpu] [--job_suffix <suffix>]
```

**Examples:**
```bash
spd-run --experiments tms_5-2 --sweep --n_agents 4            # Run TMS 5-2 sweep with 4 GPU agents
spd-run --experiments resid_mlp2 --sweep --n_agents 3 --cpu   # Run ResidMLP2 sweep with 3 CPU agents
spd-run --sweep --n_agents 10                                 # Sweep all experiments with 10 agents
spd-run --experiments tms_5-2 --sweep custom.yaml --n_agents 2 # Use custom sweep params file
```

**Sweep parameters:**
- Default sweep parameters are loaded from `spd/scripts/sweep_params.yaml`
- You can specify a custom sweep parameters file by passing its path to `--sweep`
- Sweep parameters support both experiment-specific and global configurations:
  ```yaml
  # Global parameters applied to all experiments
  global:
    seed:
      values: [0, 1, 2]
    lr:
      values: [0.001, 0.01]
  
  # Experiment-specific parameters (override global)
  tms_5-2:
    seed:
      values: [100, 200]  # Overrides global seed
    task_config:
      feature_probability:
        values: [0.05, 0.1]
  ```

#### Evaluation Runs
To run SPD with just the default hyperparameters for each experiment, use:
```bash
spd-run                                                    # Run all experiments
spd-run --experiments tms_5-2-id,resid_mlp2,resid_mlp3     # Run specific experiments
```

When multiple experiments are run without `--sweep`, it creates a W&B report with aggregated
visualizations across all experiments.

#### Additional Options
```bash
spd-run --project my-project                 # Use custom W&B project
spd-run --job_suffix test                    # Add suffix to SLURM job names
spd-run --no-create_report                   # Skip W&B report creation
```

## Development

Suggested extensions and settings for VSCode/Cursor are provided in `.vscode/`. To use the suggested
settings, copy `.vscode/settings-example.json` to `.vscode/settings.json`.

### Contributing

Please read our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project, including pull request requirements and review processes.

### Custom Metrics and Visualizations

The framework supports user-defined metrics and visualizations through `spd/user_metrics_and_figs.py`. This file is automatically created from a template during installation and provides two main functions:

- `compute_user_metrics()` - Define custom metrics logged during SPD optimization
- `create_user_figures()` - Create custom matplotlib figures during optimization

These metrics will be logged to a local file as well as wandb. You can modify this file to add your own experiment-specific metrics and visualizations without changing the core framework code.

### Development Commands

There are various `make` commands that may be helpful.

```bash
make check  # Run pre-commit on all files (i.e. basedpyright, ruff linter, and ruff formatter)
make type  # Run basedpyright on all files
make format  # Run ruff linter and formatter on all files
make test  # Run tests that aren't marked `slow`
make test-all  # Run all tests
```
