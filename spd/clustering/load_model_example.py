# %% Import dependencies
import torch
from torch.utils.data import DataLoader

from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.models.component_model import ComponentModel
from spd.utils.component_utils import component_activation_statistics

# %% Set device and disable gradients
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
print(f"Using device: {DEVICE}")

# %% Load pretrained ResidMLP model decomp from WandB
component_model, cfg, path = ComponentModel.from_pretrained("wandb:goodfire/spd/runs/0wff20d9")

# %% Create dataset compatible with the model
print("Creating dataset...")
dataset = ResidualMLPDataset(
    n_features=component_model.config.n_features,
    feature_probability=0.01,
    device=DEVICE,
    calc_labels=False,
    label_type=None,
    act_fn_name=None,
    label_fn_seed=None,
    label_coeffs=None,
    synced_inputs=component_model.get("synced_inputs", None),
)

dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
print(f"Dataset created with {component_model.config.n_features} features")

# %% Run component activation statistics
print("Computing component activation statistics...")
n_steps = 10  # Number of batches to analyze
threshold = 0.1  # Activation threshold

mean_active_per_token, mean_activation_counts = component_activation_statistics(
    model=model,
    dataloader=dataloader,
    n_steps=n_steps,
    device=str(DEVICE),
    threshold=threshold,
)

# %% Display results
print("\nComponent Activation Statistics:")
print("=" * 50)
for module_name, mean_active in mean_active_per_token.items():
    print(f"{module_name}: {mean_active:.2f} components active per token on average")

print("\nActivation counts shape for each module:")
for module_name, counts in mean_activation_counts.items():
    print(f"{module_name}: {counts.shape} (mean activation per component)")
    print(f"  Top 5 most active components: {counts.topk(5).indices.tolist()}")
    print(f"  Their activation rates: {counts.topk(5).values.tolist()}")
