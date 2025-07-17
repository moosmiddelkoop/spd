# %%
from spd.models.component_model import ComponentModel

component_model, cfg, path = ComponentModel.from_pretrained("wandb:goodfire/spd/runs/dcjm9g2n")

# %%
from muutils.dbg import dbg, dbg_auto

dbg_auto(component_model)
dbg_auto(cfg)
dbg_auto(path)

# %%
model_path = "wandb:goodfire/spd/runs/0wff20d9"
mlp_decomp_paths = [
    "wandb:spd-resid-mlp/runs/xh0qlbkj",  # 1e-6
    "wandb:spd-resid-mlp/runs/kkpzirac",  # 3e-6
    "wandb:spd-resid-mlp/runs/ziro93xq",  # Best. 1e-5
    "wandb:spd-resid-mlp/runs/pnxu3d22",  # 1e-4
    "wandb:spd-resid-mlp/runs/aahzg3zu",  # 1e-3
]

tms_decomp_paths = [
    "wandb:spd-tms/runs/f63itpo1",
    "wandb:spd-tms/runs/8bxfjeu5",
    "wandb:spd-tms/runs/xq1ivc6b",
    "wandb:spd-tms/runs/xyq22lbc",
]

component_model, cfg, path = ComponentModel.from_pretrained(tms_decomp_paths[0])


# %% Set device and disable gradients

# import torch
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_grad_enabled(False)
# print(f"Using device: {DEVICE}")

# #%% Load pretrained ResidMLP model decomp from WandB
