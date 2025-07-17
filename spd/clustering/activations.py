from typing import Any, cast

import matplotlib.pyplot as plt
import torch
from jaxtyping import Float, Int
from muutils.dbg import dbg, dbg_auto
from torch import Tensor
from torch.utils.data import DataLoader

from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, GateMLP, LinearComponent, VectorGateMLP
from spd.utils.component_utils import calc_causal_importances
from spd.utils.general_utils import extract_batch_data


def add_component_labeling(ax, component_labels: list[str], axis: str = 'x', highlight_modules: bool = True):
	"""Add component labeling and module highlighting to an axis.
	
	Args:
		ax: Matplotlib axis to modify
		component_labels: List of component labels in format "module:index"
		axis: Which axis to label ('x' or 'y')
		highlight_modules: Whether to add colored background highlighting for modules
	"""
	if not component_labels:
		return
		
	# Extract module information
	module_changes = []
	current_module = component_labels[0].split(':')[0]
	module_labels = []
	
	for i, label in enumerate(component_labels):
		module = label.split(':')[0]
		if module != current_module:
			module_changes.append(i)
			module_labels.append(current_module)
			current_module = module
	module_labels.append(current_module)
	
	# Colors for alternating module backgrounds
	colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
	
	# Add colored background regions if requested
	if highlight_modules:
		prev_idx = 0
		for i, change_idx in enumerate(module_changes + [len(component_labels)]):
			color = colors[i % len(colors)]
			if axis == 'x':
				ax.axvspan(prev_idx - 0.5, change_idx - 0.5, alpha=0.2, color=color)
			else:
				ax.axhspan(prev_idx - 0.5, change_idx - 0.5, alpha=0.2, color=color)
			prev_idx = change_idx
	
	# Add module labels
	prev_idx = 0
	for i, (change_idx, module) in enumerate(zip(module_changes + [len(component_labels)], module_labels, strict=False)):
		mid_idx = (prev_idx + change_idx) / 2
		if axis == 'x':
			ax.text(mid_idx, -0.05, module, transform=ax.get_xaxis_transform(),
					ha='center', va='top', fontsize=8, rotation=45, 
					bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
		else:
			ax.text(-0.05, mid_idx, module, transform=ax.get_yaxis_transform(),
					ha='right', va='center', fontsize=8,
					bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
		prev_idx = change_idx
	
	# Set tick labels if we want to show individual component names
	# For now, we'll just use indices but could extend this to show component names if needed
	if axis == 'x':
		ax.set_xlim(-0.5, len(component_labels) - 0.5)
	else:
		ax.set_ylim(-0.5, len(component_labels) - 0.5)


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
	save_pdf: bool = False,
	pdf_prefix: str = "activations",
	figsize_raw: tuple[int, int] = (12, 4),
	figsize_concat: tuple[int, int] = (12, 2),
	figsize_coact: tuple[int, int] = (8, 6),
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
		fig1, axs_act = plt.subplots(len(activations), 1, figsize=figsize_raw)
		if len(activations) == 1:
			axs_act = [axs_act]
		for i, (key, act) in enumerate(activations.items()):
			axs_act[i].imshow(act.T.cpu().numpy(), aspect="auto")
			axs_act[i].set_ylabel(f"components\n{key}")

		# concatenated activations
		fig2, ax2 = plt.subplots(figsize=figsize_concat)
		im2 = ax2.imshow(act_concat.T.cpu().numpy(), aspect="auto")
		ax2.set_title("Concatenated Activations")
		
		# Add component labeling on y-axis
		add_component_labeling(ax2, labels, axis='y')
		
		plt.colorbar(im2)
		
		if save_pdf:
			fig2.savefig(f"{pdf_prefix}_concatenated.pdf", bbox_inches='tight', dpi=300)

		# coactivations
		fig3, ax3 = plt.subplots(figsize=figsize_coact)
		im3 = ax3.imshow(coact.cpu().numpy(), aspect="auto")
		ax3.set_title("Coactivations")
		
		# Add component labeling on both axes
		add_component_labeling(ax3, labels, axis='x')
		add_component_labeling(ax3, labels, axis='y')
		
		plt.colorbar(im3)
		
		if save_pdf:
			fig3.savefig(f"{pdf_prefix}_coactivations.pdf", bbox_inches='tight', dpi=300)

	return output
