# SPD - Stochastic Parameter Decomposition
Code used in the paper [Stochastic Parameter Decomposition (forthcoming)](TODO)

Weights and Bias [report](https://wandb.ai/goodfire/spd-tms/reports/SPD-paper-report--VmlldzoxMzE3NzU0MQ?accessToken=427spmsbxig5cyp4jsprg9p183tysclk7ttzyxjlsiwafh8badzlpgxcvopsormm) accompanying the paper.

## Installation
From the root of the repository, run one of

```bash
make install-dev  # To install the package, dev requirements and pre-commit hooks
make install  # To just install the package (runs `pip install -e .`)
```

## Usage
Place your wandb information in a .env file. See .env.example for an example.

The repository consists of several `experiments`, each of which containing scripts to run SPD,
analyse results, and optionally a train a target model:
- `spd/experiments/tms` - Toy model of superposition
- `spd/experiments/resid_mlp` - Toy model of compressed computation and toy model of distributed
  representations
- `spd/experiments/lm` - Language model loaded from huggingface.

Note that the `lm` experiment allows for running SPD on any model pulled from huggingface, provided
you only need to decompose nn.Linear or nn.Embedding layers (other layer types would need to be
added).

### Run SPD

#### Individual Experiments
SPD can be run by executing any of the `*_decomposition.py` scripts defined in the experiment
subdirectories. Each experiment has individual config files for different variants:

**TMS (Toy Model of Superposition):**
```bash
python spd/experiments/tms/tms_decomposition.py spd/experiments/tms/tms_5-2_config.yaml
python spd/experiments/tms/tms_decomposition.py spd/experiments/tms/tms_5-2-id_config.yaml
python spd/experiments/tms/tms_decomposition.py spd/experiments/tms/tms_40-10_config.yaml
python spd/experiments/tms/tms_decomposition.py spd/experiments/tms/tms_40-10-id_config.yaml
```

**ResidMLP (Residual MLP):**
```bash
python spd/experiments/resid_mlp/resid_mlp_decomposition.py spd/experiments/resid_mlp/resid_mlp1_config.yaml
python spd/experiments/resid_mlp/resid_mlp_decomposition.py spd/experiments/resid_mlp/resid_mlp2_config.yaml
python spd/experiments/resid_mlp/resid_mlp_decomposition.py spd/experiments/resid_mlp/resid_mlp3_config.yaml
```

**Language Model:**
```bash
python spd/experiments/lm/lm_decomposition.py spd/experiments/lm/ss_emb_config.yaml
```

#### Parameter Sweeps
For running parameter sweeps on a SLURM cluster, use the sweep script:

```bash
./sweeps/sweep.sh --experiment <experiment_name> --agents <n> [--cpu] [--job_suffix <suffix>]
```

**Available experiments:**
- `tms_5-2` - TMS with 5 features, 2 hidden dimensions
- `tms_5-2-id` - TMS with 5 features, 2 hidden dimensions (fixed identity in-between)
- `tms_40-10` - TMS with 40 features, 10 hidden dimensions  
- `tms_40-10-id` - TMS with 40 features, 10 hidden dimensions (fixed identity in-between)
- `resid_mlp1` - ResidMLP with 1 layer
- `resid_mlp2` - ResidMLP with 2 layers
- `resid_mlp3` - ResidMLP with 3 layers
- `ss_emb` - SimpleStories embedding decomposition

**Examples:**
```bash
# Run TMS 5-2 sweep with 4 GPU agents
./sweeps/sweep.sh --experiment tms_5-2 --agents 4

# Run ResidMLP2 sweep with 3 CPU agents
./sweeps/sweep.sh --experiment resid_mlp2 --agents 3 --cpu

# Run with custom job suffix for identification
./sweeps/sweep.sh --experiment ss_emb --agents 2 --job_suffix my-run-123
```

The sweep script will create a WandB sweep and submit SLURM jobs to run multiple agents in parallel. Output logs can be found in `~/slurm_logs/slurm-<job_id>.out`.

All experiments call the `optimize` function in `spd/run_spd.py`, which contains the main SPD logic.

## Development

Suggested extensions and settings for VSCode/Cursor are provided in `.vscode/`. To use the suggested
settings, copy `.vscode/settings-example.json` to `.vscode/settings.json`.

There are various `make` commands that may be helpful

```bash
make check  # Run pre-commit on all files (i.e. pyright, ruff linter, and ruff formatter)
make type  # Run pyright on all files
make format  # Run ruff linter and formatter on all files
make test  # Run tests that aren't marked `slow`
make test-all  # Run all tests
```