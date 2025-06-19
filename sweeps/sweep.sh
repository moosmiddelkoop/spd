#!/bin/bash

# Script for running sweeps on a slurm cluster
#
# USAGE GUIDE:
# This script creates a Weights & Biases (wandb) sweep and deploys multiple SLURM agents to run it.
# 
# Basic usage:
#   ./sweep.sh --experiment tms_5-2-id --agents 5
#   ./sweep.sh -e resid_mlp2 -n 10 --cpu
#   ./sweep.sh -e ss_emb -n 3 -s my-run-123
#
# EXPERIMENTS:
#   tms_5-2      - TMS with 5 features, 2 hidden dimensions
#   tms_5-2-id   - TMS with 5 features, 2 hidden dimensions (fixed identity in-between)
#   tms_40-10    - TMS with 40 features, 10 hidden dimensions
#   tms_40-10-id - TMS with 40 features, 10 hidden dimensions (fixed identity in-between)
#   resid_mlp1   - ResidMLP with 1 layer
#   resid_mlp2   - ResidMLP with 2 layers
#   resid_mlp3   - ResidMLP with 3 layers
#   ss_emb       - SimpleStories embedding decomposition
#
# OPTIONS:
#   -e, --experiment    Required. Specific experiment to run (see EXPERIMENTS list above)
#   -n, --agents        Required. Number of SLURM agents to deploy for the sweep
#   -c, --cpu           Optional. Run on CPU instead of GPU (default: use GPU)
#   -s, --job_suffix    Optional. Suffix to add to SLURM job names for identification
#
# The script will:
# 1. Create a wandb sweep using the experiment's sweep config
# 2. Submit N identical SLURM jobs that each run a wandb agent
# 3. Output the sweep URL for monitoring progress

set -euo pipefail

# Set the SPD repository path below
SPD_REPO="$HOME/spd"

###############################################################################
# Argument parsing                                                            #
###############################################################################
experiment=""
n_agents=0
use_cpu=false
job_suffix=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--experiment)
            experiment="$2"
            shift 2
            ;;
        -n|--agents)
            n_agents="$2"
            shift 2
            ;;
        -c|--cpu)
            use_cpu=true
            shift
            ;;
        -s|--job_suffix)
            job_suffix="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 -e|--experiment <experiment_name> -n|--agents <n> [-c|--cpu] [-s|--job_suffix <suffix>]"
            echo ""
            echo "Create a wandb sweep and deploy SLURM agents to run it."
            echo ""
            echo "Required arguments:"
            echo "  -e, --experiment    Experiment name: tms_5-2, tms_5-2-id, tms_40-10, tms_40-10-id, resid_mlp1, resid_mlp2, resid_mlp3, ss_emb"
            echo "  -n, --agents        Number of SLURM agents to deploy"
            echo ""
            echo "Optional arguments:"
            echo "  -c, --cpu           Use CPU instead of GPU (default: use GPU)"
            echo "  -s, --job_suffix    Suffix for SLURM job names"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 -e tms_5-2 -n 5                 # Run TMS 5-2 sweep with 5 GPU agents"
            echo "  $0 --experiment resid_mlp1 --agents 3 --cpu  # Run ResidMLP1 sweep with 3 CPU agents"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$experiment" ]]; then
    echo "Missing --experiment flag."
    exit 1
fi

if [[ "$n_agents" -le 0 ]]; then
    echo "Please supply a positive integer for --agents."
    exit 1
fi


###############################################################################
# Resolve project & paths for requested experiment                            #
###############################################################################
# Map experiment name to wandb project, decomposition script, and config file
case "$experiment" in
    tms_5-2)
        wandb_project="spd-tms"
        decomp_script="$SPD_REPO/spd/experiments/tms/tms_decomposition.py"
        config_path="$SPD_REPO/spd/experiments/tms/tms_5-2_config.yaml"
        ;;
    tms_5-2-id)
        wandb_project="spd-tms"
        decomp_script="$SPD_REPO/spd/experiments/tms/tms_decomposition.py"
        config_path="$SPD_REPO/spd/experiments/tms/tms_5-2-id_config.yaml"
        ;;
    tms_40-10)
        wandb_project="spd-tms"
        decomp_script="$SPD_REPO/spd/experiments/tms/tms_decomposition.py"
        config_path="$SPD_REPO/spd/experiments/tms/tms_40-10_config.yaml"
        ;;
    tms_40-10-id)
        wandb_project="spd-tms"
        decomp_script="$SPD_REPO/spd/experiments/tms/tms_decomposition.py"
        config_path="$SPD_REPO/spd/experiments/tms/tms_40-10-id_config.yaml"
        ;;
    resid_mlp1)
        wandb_project="spd-resid-mlp"
        decomp_script="$SPD_REPO/spd/experiments/resid_mlp/resid_mlp_decomposition.py"
        config_path="$SPD_REPO/spd/experiments/resid_mlp/resid_mlp1_config.yaml"
        ;;
    resid_mlp2)
        wandb_project="spd-resid-mlp"
        decomp_script="$SPD_REPO/spd/experiments/resid_mlp/resid_mlp_decomposition.py"
        config_path="$SPD_REPO/spd/experiments/resid_mlp/resid_mlp2_config.yaml"
        ;;
    resid_mlp3)
        wandb_project="spd-resid-mlp"
        decomp_script="$SPD_REPO/spd/experiments/resid_mlp/resid_mlp_decomposition.py"
        config_path="$SPD_REPO/spd/experiments/resid_mlp/resid_mlp3_config.yaml"
        ;;
    ss_emb)
        wandb_project="spd-lm"
        decomp_script="$SPD_REPO/spd/experiments/lm/lm_decomposition.py"
        config_path="$SPD_REPO/spd/experiments/lm/ss_emb_config.yaml"
        ;;
    *)
        echo "Unsupported experiment: $experiment"
        echo "Supported experiments: tms_5-2, tms_5-2-id, tms_40-10, tms_40-10-id, resid_mlp1, resid_mlp2, resid_mlp3, ss_emb"
        exit 1
        ;;
esac

###############################################################################
# Process sweep config template                                               #
###############################################################################
# Use the centralized sweep config template and substitute paths
base_sweep_cfg="$SPD_REPO/sweeps/sweep_config.yaml"
temp_sweep_cfg="/tmp/sweep_config_${experiment}_$$.yaml"

# Create temporary config file with substituted paths
sed -e "s#<path/to/_decomposition.py>#${decomp_script}#g" \
    -e "s#<path/to/config.yaml>#${config_path}#g" \
    "$base_sweep_cfg" > "$temp_sweep_cfg"

echo "Using sweep config for $experiment:"
echo "  Decomposition script: $decomp_script"
echo "  Config file: $config_path"
echo "  Sweep file: $base_sweep_cfg"

###############################################################################
# Create the sweep                                                            #
###############################################################################
# Run wandb sweep command and capture both output and exit code
if sweep_output=$(wandb sweep --project "$wandb_project" "$temp_sweep_cfg" 2>&1); then
    wandb_exit_code=0
else
    wandb_exit_code=$?
fi

if [[ $wandb_exit_code -ne 0 ]]; then
    echo "wandb sweep failed (exit $wandb_exit_code):"
    echo "$sweep_output"
    rm -f "$temp_sweep_cfg"  # Clean up temp file
    exit 1
fi

# Clean up temporary config file
rm -f "$temp_sweep_cfg"

# Extract the agent ID from wandb output (format: org/project/sweep_id)
agent_id=$(echo "$sweep_output" | grep "wandb agent" | awk '{print $NF}')
[[ -z "$agent_id" ]] && { echo "Could not extract agent ID"; exit 1; }

# Parse agent ID components and construct the sweep URL
IFS='/' read -r org_name project_name sweep_id <<< "$agent_id"
wandb_url="https://wandb.ai/${org_name}/${project_name}/sweeps/${sweep_id}"
echo "Sweep created: $wandb_url"
echo "Deploying $n_agents agents for experiment $experiment..."

###############################################################################
# Fire up agents                                                              #
###############################################################################

# Set GPU or CPU configuration based on --cpu flag
if [ "$use_cpu" = true ]; then
    gpu_config="#SBATCH --gres=gpu:0"
else
    gpu_config="#SBATCH --gres=gpu:1"
fi

# Set job name with optional suffix
job_name="spd-sweep"
if [[ -n "$job_suffix" ]]; then
    job_name="${job_name}-${job_suffix}"
fi

# Make sure SLURM logs directory exists
mkdir -p "$HOME/slurm_logs"

# Create or update the run_agent.sh script in the home directory
# This script will be submitted to SLURM multiple times
cat > "$HOME/run_agent.sh" << EOL
#!/bin/bash
#SBATCH --nodes=1
$gpu_config
#SBATCH --time=24:00:00
#SBATCH --job-name=${job_name}
#SBATCH --partition=all
#SBATCH --output=$HOME/slurm_logs/slurm-%j.out

# Change to the SPD repository directory
cd $SPD_REPO

# This is the actual command that runs in each SLURM job
wandb agent $agent_id
EOL

chmod +x "$HOME/run_agent.sh"

# Submit the job n times to create n parallel agents
for ((i=1; i<=$n_agents; i++)); do
    sbatch "$HOME/run_agent.sh"
done

echo "All $n_agents agents deployed. Sweep URL: $wandb_url"
