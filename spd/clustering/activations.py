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


def plot_activations(
	activations: dict[str, Float[Tensor, " n_steps C"]],
	act_concat: Float[Tensor, " n_steps c"],
	coact: Float[Tensor, " c c"],
	labels: list[str],
	save_pdf: bool = False,
	pdf_prefix: str = "activations",
	figsize_raw: tuple[int, int] = (12, 4),
	figsize_concat: tuple[int, int] = (12, 2),
	figsize_coact: tuple[int, int] = (8, 6),
) -> None:
	"""Plot activation visualizations including raw, concatenated, sorted, and coactivations.
	
	Args:
		activations: Dictionary of raw activations by module
		act_concat: Concatenated activations tensor
		coact: Coactivation matrix
		labels: Component labels
		save_pdf: Whether to save plots as PDFs
		pdf_prefix: Prefix for PDF filenames
		figsize_raw: Figure size for raw activations
		figsize_concat: Figure size for concatenated activations
		figsize_coact: Figure size for coactivations
	"""
	# Raw activations
	fig1, axs_act = plt.subplots(len(activations), 1, figsize=figsize_raw)
	if len(activations) == 1:
		axs_act = [axs_act]
	for i, (key, act) in enumerate(activations.items()):
		axs_act[i].matshow(act.T.cpu().numpy(), aspect="auto")
		axs_act[i].set_ylabel(f"components\n{key}")

	# Concatenated activations
	fig2, ax2 = plt.subplots(figsize=figsize_concat)
	im2 = ax2.matshow(act_concat.T.cpu().numpy(), aspect="auto")
	ax2.set_title("Concatenated Activations")
	
	# Add component labeling on y-axis
	add_component_labeling(ax2, labels, axis='y')
	
	plt.colorbar(im2)
	
	if save_pdf:
		fig2.savefig(f"{pdf_prefix}_concatenated.pdf", bbox_inches='tight', dpi=300)

	# Concatenated activations, sorted samples
	fig3, ax3 = plt.subplots(figsize=figsize_concat)
	
	# Compute gram matrix (sample similarity) and sort samples
	gram_matrix: Float[Tensor, " n_steps n_steps"] = act_concat @ act_concat.T
	
	# Use hierarchical clustering approach for better sorting
	similarity_scores: Float[Tensor, " n_steps"] = gram_matrix.sum(dim=1)
	sorted_indices: torch.Tensor = torch.argsort(similarity_scores, descending=True)
	
	act_concat_sorted: Float[Tensor, " n_steps c"] = act_concat[sorted_indices]
	
	im3 = ax3.matshow(torch.log10(act_concat_sorted).T.cpu().numpy(), aspect="auto")
	ax3.set_title("Concatenated Activations $\\log_{10}$, Sorted Samples")
	
	# Add component labeling on y-axis
	add_component_labeling(ax3, labels, axis='y')
	
	plt.colorbar(im3)
	
	if save_pdf:
		fig3.savefig(f"{pdf_prefix}_concatenated_sorted.pdf", bbox_inches='tight', dpi=300)

	# Coactivations
	fig4, ax4 = plt.subplots(figsize=figsize_coact)
	im4 = ax4.matshow(coact.cpu().numpy(), aspect="auto")
	ax4.set_title("Coactivations")
	
	# Add component labeling on both axes
	add_component_labeling(ax4, labels, axis='x')
	add_component_labeling(ax4, labels, axis='y')
	
	plt.colorbar(im4)
	
	if save_pdf:
		fig4.savefig(f"{pdf_prefix}_coactivations.pdf", bbox_inches='tight', dpi=300)


def add_component_labeling(ax: plt.Axes, component_labels: list[str], axis: str = 'x') -> None:
	"""Add component labeling using major/minor ticks to show module boundaries.
	
	Args:
		ax: Matplotlib axis to modify
		component_labels: List of component labels in format "module:index"
		axis: Which axis to label ('x' or 'y')
	"""
	if not component_labels:
		return
		
	# Extract module information
	module_changes: list[int] = []
	current_module: str = component_labels[0].split(':')[0]
	module_labels: list[str] = []
	
	for i, label in enumerate(component_labels):
		module: str = label.split(':')[0]
		if module != current_module:
			module_changes.append(i)
			module_labels.append(current_module)
			current_module = module
	module_labels.append(current_module)
	
	# Set up major and minor ticks
	# Minor ticks: every 10 components
	minor_ticks: list[int] = list(range(0, len(component_labels), 10))
	
	# Major ticks: module boundaries (start of each module)
	major_ticks: list[int] = [0] + module_changes
	major_labels: list[str] = module_labels
	
	if axis == 'x':
		ax.set_xticks(minor_ticks, minor=True)
		ax.set_xticks(major_ticks)
		ax.set_xticklabels(major_labels)
		ax.set_xlim(-0.5, len(component_labels) - 0.5)
		# Style the ticks
		ax.tick_params(axis='x', which='minor', length=2, width=0.5)
		ax.tick_params(axis='x', which='major', length=6, width=1.5)
	else:
		ax.set_yticks(minor_ticks, minor=True)
		ax.set_yticks(major_ticks)
		ax.set_yticklabels(major_labels)
		ax.set_ylim(-0.5, len(component_labels) - 0.5)
		# Style the ticks
		ax.tick_params(axis='y', which='minor', length=2, width=0.5)
		ax.tick_params(axis='y', which='major', length=6, width=1.5)


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
		plot_activations(
			activations=activations,
			act_concat=act_concat,
			coact=coact,
			labels=labels,
			save_pdf=save_pdf,
			pdf_prefix=pdf_prefix,
			figsize_raw=figsize_raw,
			figsize_concat=figsize_concat,
			figsize_coact=figsize_coact,
		)

	return output
