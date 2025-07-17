import random
from collections.abc import Callable
from typing import Any, cast
import warnings

import matplotlib.pyplot as plt
import torch
from jaxtyping import Bool, Float, Int
from muutils.dbg import dbg, dbg_auto
from torch import Tensor
from torch.utils.data import DataLoader

from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, GateMLP, LinearComponent, VectorGateMLP
from spd.utils.component_utils import calc_causal_importances
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.general_utils import extract_batch_data
from spd.clustering.merge_matrix import GroupMerge

@torch.no_grad()
def component_activations(
    model: ComponentModel,
    dataloader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    device: torch.device,
) -> dict[str, Float[Tensor, " n_steps C"]]:
    """Get the number and strength of the masks over the full dataset."""
    # We used "-" instead of "." as module names can't have "." in them
    gates: dict[str, GateMLP | VectorGateMLP] = {
        k.removeprefix("gates.").replace("-", "."): cast(GateMLP | VectorGateMLP, v)
        for k, v in model.gates.items()
    }
    components: dict[str, LinearComponent | EmbeddingComponent] = {
        k.removeprefix("components.").replace("-", "."): cast(
            LinearComponent | EmbeddingComponent, v
        )
        for k, v in model.components.items()
    }

    # --- Get Batch --- #
    batch = extract_batch_data(next(iter(dataloader)))
    batch = batch.to(device)

    _, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
        batch, module_names=list(components.keys())
    )
    Vs = {module_name: v.V for module_name, v in components.items()}

    causal_importances, _ = calc_causal_importances(
        pre_weight_acts=pre_weight_acts,
        Vs=Vs,
        gates=gates,
        detach_inputs=False,
    )

    return causal_importances




def process_activations(
	activations: dict[str, Float[Tensor, " n_steps C"]], # module name to sample x component gate activations
	filter_dead_threshold: float = 0.01,
	plots: bool = False,
) -> dict[str, Any]:
	"""get back a dict of coactivations, slices, and concated activations"""
	
	# compute the labels and total component count
	total_c: int = 0
	labels: list[str] = list()
	for key, act in activations.items():
		c = act.shape[-1]
		labels.extend([f"{key}:{i}" for i in range(c)])		
		total_c += c

	dbg(total_c)

	# concat the activations
	act_concat: Float[Tensor, " n_steps c"] = torch.cat(
		[activations[key] for key in activations], dim=-1
	)

	# filter dead components
	dead_components_lst: list[str]| None = None
	if filter_dead_threshold > 0:
		dead_components_lst = list()
		max_act = act_concat.max(dim=0).values
		dead_components = max_act < filter_dead_threshold
		if dead_components.any():
			act_concat = act_concat[:, ~dead_components]
			alive_labels: list[tuple[str, bool]] = [
				(lbl, keep.item())
				for lbl, keep in zip(labels, ~dead_components, strict=False)
			]
			labels = [label for label, keep in alive_labels if keep]		
			dead_components_lst = [label for label, keep in alive_labels if not keep]
			dbg((len(dead_components_lst), len(labels)))

	# compute coactivations
	coact: Float[Tensor, " c c"] = act_concat.T @ act_concat / act_concat.shape[0]

	# return the output
	output: dict[str, Any] = dict(
		activations=act_concat,
		labels=labels,
		coactivations=coact,
		dead_components_lst=dead_components_lst,
		n_components_original=total_c,
		n_components_alive=len(labels),
		n_components_dead=len(dead_components_lst) if dead_components_lst else 0,
	)

	dbg_auto(output)

	if plots:
		# raw activations
		_, axs_act = plt.subplots(len(activations), 1, figsize=(12, 4))
		for i, (key, act) in enumerate(activations.items()):
			axs_act[i].imshow(act.T.cpu().numpy(), aspect="auto")
			axs_act[i].set_ylabel(f"components\n{key}")

		# concatenated activations
		plt.figure(figsize=(12, 2))
		plt.imshow(act_concat.T.cpu().numpy(), aspect="auto")
		plt.title("Concatenated Activations")
		plt.colorbar()

		# coactivations
		plt.figure(figsize=(8, 6))
		plt.imshow(coact.cpu().numpy(), aspect="auto")
		plt.title("Coactivations")
		plt.colorbar()

	return output
