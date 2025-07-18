# %%
import torch
from spd.models.component_model import ComponentModel
from muutils.dbg import dbg_auto
component_model, cfg, path = ComponentModel.from_pretrained("data/model_50	000.pth")

