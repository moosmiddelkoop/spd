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
subdirectories, along with a corresponding config file. E.g.
```bash
python spd/experiments/tms/tms_decomposition.py spd/experiments/tms/tms_5-2_config.yaml
```

Each experiment is also saved in a registry which is used for sweeps and evals.

**Available experiments** (defined in `spd/registry.py`):
- `tms_5-2` - TMS with 5 features, 2 hidden dimensions
- `tms_5-2-id` - TMS with 5 features, 2 hidden dimensions (fixed identity in-between)
- `tms_40-10` - TMS with 40 features, 10 hidden dimensions  
- `tms_40-10-id` - TMS with 40 features, 10 hidden dimensions (fixed identity in-between)
- `resid_mlp1` - ResidMLP with 1 layer
- `resid_mlp2` - ResidMLP with 2 layers
- `resid_mlp3` - ResidMLP with 3 layers
- `ss_emb` - SimpleStories embedding decomposition

#### Sweeps
For running sweeps on a SLURM cluster, set your desired sweep parameters in
`spd/sweeps/sweep_params.yaml` and then call:

```bash
spd-sweep <experiment_name> <n_agents> [--cpu] [--job_suffix <suffix>]
```
where <experiment_name> comes from a key in the registry (only experiments registered here
can be run in sweeps).

**Examples:**
```bash
spd-sweep tms_5-2 4 --job_suffix 5m             # Run TMS 5-2 sweep with 4 GPU agents
spd-sweep resid_mlp2 3 --cpu                    # Run ResidMLP2 sweep with 3 CPU agents
spd-sweep ss_emb 2 --job_suffix 1h              # Run with custom job suffix
```

(Note, the `spd-sweep` command will call `spd/sweeps/sweep.py`).

#### Evals
To test your changes on all experiments in the registry, run:
```bash
spd-evals                                                    # Run all experiments
spd-evals --experiments tms_5-2-id,resid_mlp2,resid_mlp3     # Run only the experiments listed
```
This will deploy a slurm job for each experiment.

(Note, the `spd-sweep` command will call `spd/sweeps/sweep.py`).

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