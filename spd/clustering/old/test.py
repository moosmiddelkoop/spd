# %%
import torch
from muutils.dbg import dbg_auto

from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.models.component_model import ComponentModel
from spd.utils.data_utils import DatasetGeneratedDataLoader

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
component_model, cfg, path = ComponentModel.from_pretrained("wandb:goodfire/spd/runs/dcjm9g2n")

dbg_auto(component_model)
dbg_auto(cfg)
dbg_auto(path)
dir(component_model)

# %%
dataset = ResidualMLPDataset(
    n_features=component_model.config.n_features,
    feature_probability=cfg.task_config.feature_probability,
    device=DEVICE,
    calc_labels=False,  # Our labels will be the output of the target model
    label_type=None,
    act_fn_name=None,
    label_fn_seed=None,
    label_coeffs=None,
    data_generation_type=cfg.task_config.data_generation_type,
    # synced_inputs=synced_inputs,
)

train_loader = DatasetGeneratedDataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)


# %% Set device and disable gradients

# import torch
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_grad_enabled(False)
# print(f"Using device: {DEVICE}")

# #%% Load pretrained ResidMLP model decomp from WandB
