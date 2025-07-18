import random
import warnings
from typing import Callable, Any

import matplotlib.pyplot as plt
import torch
from jaxtyping import Bool, Float
from torch import Tensor

from spd.clustering.merge_matrix import GroupMerge


def format_scientific_latex(value: float) -> str:
	"""Format a number in LaTeX scientific notation style."""
	if value == 0:
		return r"$0$"
	
	import math
	exponent: int = int(math.floor(math.log10(abs(value))))
	mantissa: float = value / (10 ** exponent)
	
	return f"${mantissa:.2f} \\times 10^{{{exponent}}}$"


def plot_merge_iteration(
	current_merge: GroupMerge,
	current_coact: Float[Tensor, "k_groups k_groups"],
	costs: Float[Tensor, "k_groups k_groups"],
	min_pair: tuple[int, int],
	pair_cost: float,
	iteration: int,
	component_labels: list[str] | None = None,
	figsize: tuple[int, int] = (16, 3),
	save_pdf: bool = False,
	pdf_prefix: str = "merge_iteration",
	tick_spacing: int = 10,
) -> None:
	"""Plot merge iteration results with merge tree, coactivations, and costs.
	
	Args:
		current_merge: Current merge state
		current_coact: Current coactivation matrix
		costs: Current cost matrix
		min_pair: Selected merge pair indices
		pair_cost: Cost of selected merge pair
		iteration: Current iteration number
		component_labels: Component labels for axis labeling
		figsize: Figure size
		save_pdf: Whether to save as PDF
		pdf_prefix: Prefix for PDF filename
		tick_spacing: Spacing for minor ticks
	"""
	fig, axs = plt.subplots(
		1, 3,
		figsize=figsize,
		sharey=True,
		gridspec_kw={"width_ratios": [2, 1, 1]}
	)
	
	# Merge plot
	current_merge.plot(ax=axs[0], show=False, component_labels=component_labels)
	axs[0].set_title("Merge")
	
	# Coactivations plot
	axs[1].matshow(current_coact.cpu().numpy(), aspect='equal')
	coact_min: float = current_coact.min().item()
	coact_max: float = current_coact.max().item()
	coact_min_str: str = format_scientific_latex(coact_min)
	coact_max_str: str = format_scientific_latex(coact_max)
	axs[1].set_title(f"Coactivations\n[{coact_min_str}, {coact_max_str}]")
	
	# Setup ticks for coactivations
	k_groups: int = current_coact.shape[0]
	minor_ticks: list[int] = list(range(0, k_groups, tick_spacing))
	axs[1].set_yticks(minor_ticks)
	axs[1].set_xticks(minor_ticks)
	axs[1].set_xticklabels([])  # Remove x-axis tick labels but keep ticks
	
	# Costs plot
	axs[2].matshow(costs.cpu().numpy(), aspect='equal')
	costs_min: float = costs.min().item()
	costs_max: float = costs.max().item()
	costs_min_str: str = format_scientific_latex(costs_min)
	costs_max_str: str = format_scientific_latex(costs_max)
	axs[2].set_title(f"Costs\n[{costs_min_str}, {costs_max_str}]")
	
	# Setup ticks for costs
	axs[2].set_yticks(minor_ticks)
	axs[2].set_xticks(minor_ticks)
	axs[2].set_xticklabels([])  # Remove x-axis tick labels but keep ticks
	
	fig.suptitle(f"Iteration {iteration} with cost {pair_cost:.4f}")
	plt.tight_layout()
	
	if save_pdf:
		fig.savefig(f"{pdf_prefix}_iter_{iteration:03d}.pdf", bbox_inches='tight', dpi=300)
	
	plt.show()


def compute_merge_costs(
    coact: Bool[Tensor, "k_groups k_groups"],
    merges: GroupMerge,
    alpha: float = 1.0,
    # rank_cost: Callable[[float], float] = lambda c: math.log(c),
    rank_cost: Callable[[float], float] = lambda _: 1.0,
) -> Float[Tensor, "k_groups k_groups"]:
    """Compute MDL costs for merge matrices"""
    device: torch.device = coact.device
    ranks: Float[Tensor, " k_groups"] = merges.components_per_group.to(device=device).float()
    diag: Float[Tensor, " k_groups"] = torch.diag(coact).to(device=device)

    # dbg_tensor(coact)
    # dbg_tensor(ranks)
    # dbg_tensor(diag)

    return alpha * (
        diag @ ranks.T
        + ranks @ diag.T
        - (
            ranks.unsqueeze(0) 
            + ranks.unsqueeze(1)
            + (rank_cost(merges.k_groups) / alpha)
        ) * coact
    )



def recompute_coacts(
	coact: Float[Tensor, "k_groups k_groups"],
	merges: GroupMerge,
	merge_pair: tuple[int, int],
	activation_mask: Bool[Tensor, "samples k_groups"],
) -> tuple[
		GroupMerge,
		Float[Tensor, "k_groups-1 k_groups-1"],
		Bool[Tensor, "samples k_groups"],
	]:
	# check shape
	k_groups: int = coact.shape[0]
	assert coact.shape[1] == k_groups, "Coactivation matrix must be square"

	# activations of the new merged group
	activation_mask_grp: Bool[Tensor, " samples"] = activation_mask[:, merge_pair[0]] + activation_mask[:, merge_pair[1]]

	# coactivations with the new merged group
	# dbg_tensor(activation_mask_grp)
	# dbg_tensor(activation_mask)
	coact_with_merge: Bool[Tensor, " k_groups"] = (activation_mask_grp.float() @ activation_mask.float()).bool()
	new_group_idx: int = min(merge_pair)
	remove_idx: int = max(merge_pair)
	new_group_self_coact: float = activation_mask_grp.float().sum().item()
	# dbg_tensor(coact_with_merge)

	# assemble the merge pair
	merge_new: GroupMerge = merges.merge_groups(
		merge_pair[0],
		merge_pair[1],
	)
	old_to_new_idx: dict[int|None, int| None] = merge_new.old_to_new_idx # type: ignore
	assert old_to_new_idx[None] == new_group_idx, "New group index should be the minimum of the merge pair"
	assert old_to_new_idx[new_group_idx] is None
	assert old_to_new_idx[remove_idx] is None
	# TODO: check that the rest are in order? probably not necessary

	# reindex coactivations
	coact_temp: Float[Tensor, "k_groups k_groups"] = coact.clone()
	# add in the similarities with the new group
	coact_temp[new_group_idx, :] = coact_with_merge
	coact_temp[:, new_group_idx] = coact_with_merge
	# delete the old group
	mask: Bool[Tensor, " k_groups"] = torch.ones(coact_temp.shape[0], dtype=torch.bool, device=coact_temp.device)
	mask[remove_idx] = False
	coact_new: Float[Tensor, "k_groups-1 k_groups-1"] = coact_temp[mask, :][:, mask]
	# add in the self-coactivation of the new group
	coact_new[new_group_idx, new_group_idx] = new_group_self_coact
	# dbg_tensor(coact_new)

	# reindex mask
	activation_mask_new: Float[Tensor, "samples ..."] = activation_mask.clone()
	# add in the new group
	activation_mask_new[:, new_group_idx] = activation_mask_grp
	# remove the old group
	activation_mask_new = activation_mask_new[:, mask]
	
	# dbg_tensor(activation_mask_new)

	return (
		merge_new,
		coact_new,
		activation_mask_new,
	)

def merge_iteration(
	coact: Bool[Tensor, "c_components c_components"],
	activation_mask: Bool[Tensor, "samples c_components"],
	initial_merge: GroupMerge|None = None,
    alpha: float = 1.0,
	iters: int = 100,
	check_threshold: float = 0.05,
	pop_component_prob: float = 0.0,
	rank_cost: Callable[[float], float] = lambda _: 1.0,
	stopping_condition: Callable[[dict[str, Any]], bool] | None = None,
	plot_every: int = 20,
	plot_every_min: int = 0,
	save_pdf: bool = False,
	pdf_prefix: str = "merge_iteration",
	component_labels: list[str] | None = None,
	figsize: tuple[int, int] = (16, 3),
	figsize_final: tuple[int, int] = (10, 6),
	tick_spacing: int = 10,
	plot_final: bool = True,
) -> dict[str, list[float] | GroupMerge]:
	# check shapes
	c_components: int = coact.shape[0]
	assert coact.shape[1] == c_components, "Coactivation matrix must be square"
	assert activation_mask.shape[1] == c_components, "Activation mask must match coactivation matrix shape"


	do_pop: bool = pop_component_prob > 0.0
	if do_pop:
		iter_pop: Bool[Tensor, " iters"] = torch.rand(iters, device=coact.device) < pop_component_prob

	# start with an identity merge
	current_merge: GroupMerge
	if initial_merge is not None:
		current_merge = initial_merge
	else:
		current_merge = GroupMerge.identity(n_components=c_components)

	k_groups: int = c_components
	current_coact: Float[Tensor, "k_groups k_groups"] = coact.clone()
	current_act_mask: Bool[Tensor, "samples k_groups"] = activation_mask.clone()


	merge_costs: dict[str, list[float]] = dict(
		non_diag_costs_min=[],
		non_diag_costs_max=[],
		max_considered_cost=[],
		selected_pair_cost=[],
	)

	# iteration counter
	i: int = 0
	while i < iters:
		# TODO: re-add popping components

		# compute costs
		costs: Float[Tensor, "c_components c_components"] = compute_merge_costs(
			coact=current_coact,
			merges=current_merge,
			alpha=alpha,
			rank_cost=rank_cost,
		)

		# find the maximum cost among non-diagonal elements we should consider
		non_diag_costs: Float[Tensor, ""] = costs[~torch.eye(k_groups, dtype=torch.bool)]
		non_diag_costs_range: tuple[float, float] = (non_diag_costs.min().item(), non_diag_costs.max().item())
		max_considered_cost: float = (non_diag_costs_range[1] - non_diag_costs_range[0]) * check_threshold + non_diag_costs_range[0]

		merge_costs['non_diag_costs_min'].append(non_diag_costs_range[0])
		merge_costs['non_diag_costs_max'].append(non_diag_costs_range[1])
		merge_costs['max_considered_cost'].append(max_considered_cost)

		# consider pairs with costs below the threshold
		considered_idxs = torch.where(costs <= max_considered_cost)
		considered_idxs = torch.stack(considered_idxs, dim=1)
		# remove from considered_idxs where i == j
		considered_idxs = considered_idxs[considered_idxs[:, 0] != considered_idxs[:, 1]]		

		# randomly select one of the considered pairs
		min_pair: tuple[int, int] = tuple(considered_idxs[random.randint(0, considered_idxs.shape[0] - 1)].tolist())
		pair_cost: float = costs[min_pair[0], min_pair[1]].item()
		
		# Track the selected pair cost
		merge_costs['selected_pair_cost'].append(pair_cost)

		# merge the pair
		current_merge, current_coact, current_act_mask = recompute_coacts(
			coact=current_coact,
			merges=current_merge,
			merge_pair=min_pair,
			activation_mask=current_act_mask,
		)

		# dbg_tensor(costs)
		# dbg_tensor(non_diag_costs)
		# dbg(non_diag_costs_range)		
		# dbg(max_considered_cost)
		# dbg_tensor(considered_idxs)
		# print(f"Iteration {i}: merging pair {min_pair=} {pair_cost=} {non_diag_costs_range[0]=} {max_considered_cost=}", flush=True)

		k_groups -= 1
		assert current_coact.shape[0] == k_groups, "Coactivation matrix shape should match number of groups"
		assert current_coact.shape[1] == k_groups, "Coactivation matrix shape should match number of groups"
		assert current_act_mask.shape[1] == k_groups, "Activation mask shape should match number of groups"

		# Check stopping conditions
		if k_groups <= 2:
			warnings.warn(f"Stopping early at iteration {i} as only {k_groups} groups left")
			current_merge.plot(component_labels=component_labels)
			if save_pdf:
				plt.savefig(f"{pdf_prefix}_final_early.pdf", bbox_inches='tight', dpi=300)
			plt.show()
			break
			
		# Custom stopping condition
		if stopping_condition is not None:
			iteration_stats = {
				'iteration': i,
				'k_groups': k_groups,
				'non_diag_costs_min': merge_costs['non_diag_costs_min'],
				'non_diag_costs_max': merge_costs['non_diag_costs_max'],
				'selected_pair_cost': merge_costs['selected_pair_cost'],
				'max_considered_cost': merge_costs['max_considered_cost'],
				'current_cost_min': non_diag_costs_range[0],
				'current_cost_max': non_diag_costs_range[1],
				'pair_cost': pair_cost,
			}
			if stopping_condition(iteration_stats):
				break

		if plot_every and (i >= plot_every_min) and (i % plot_every == 0):
			plot_merge_iteration(
				current_merge=current_merge,
				current_coact=current_coact,
				costs=costs,
				min_pair=min_pair,
				pair_cost=pair_cost,
				iteration=i,
				component_labels=component_labels,
				figsize=figsize,
				save_pdf=save_pdf,
				pdf_prefix=pdf_prefix,
				tick_spacing=tick_spacing,
			)

		i += 1


	# Final cost evolution plot
	if plot_final:
		plt.figure(figsize=figsize_final)
		plt.plot(merge_costs['max_considered_cost'], label='max considered cost')
		plt.plot(merge_costs['non_diag_costs_min'], label='non-diag costs min')
		plt.plot(merge_costs['non_diag_costs_max'], label='non-diag costs max')
		plt.plot(merge_costs['selected_pair_cost'], label='selected pair cost')
		plt.xlabel("Iteration")
		plt.ylabel("Cost")
		plt.legend()
		
		if save_pdf:
			plt.savefig(f"{pdf_prefix}_cost_evolution.pdf", bbox_inches='tight', dpi=300)
		
		plt.show()
		
	# Return results for sweep analysis
	return {
		**merge_costs,
		'final_merge': current_merge,
		'total_iterations': i,
		'final_k_groups': k_groups,
	}