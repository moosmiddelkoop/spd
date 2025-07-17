import random
import warnings
from collections.abc import Callable

import matplotlib.pyplot as plt
import torch
from jaxtyping import Bool, Float
from torch import Tensor

from spd.clustering.merge_matrix import GroupMerge


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
	plot_every: int = 20,
	plot_every_min: int = 0,
):
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

		if k_groups <= 2:
			warnings.warn(f"Stopping early at iteration {i} as only {k_groups} groups left")
			current_merge.plot()
			plt.show()
			break

		if plot_every and (i >= plot_every_min) and (i % plot_every == 0):
			# current_merge.plot((5,1))
			# plt.show()
			# # _, ax = plt.subplots(figsize=(2, 2))
			# plt.imshow(current_coact.cpu().numpy(), aspect='equal')
			# plt.colorbar()
			# plt.show()
			# plt.imshow(costs.cpu().numpy(), aspect='equal')
			# plt.colorbar()
			# plt.show()

			fig, axs = plt.subplots(
				1, 3,
				figsize=(16, 3),
				sharey=True,
				gridspec_kw={"width_ratios": [2, 1, 1]}
			)
			current_merge.plot(ax=axs[0], show=False)
			axs[0].set_title("Merge")
			axs[1].imshow(current_coact.cpu().numpy(), aspect='equal')
			axs[1].set_title(f"Coactivations\n[{current_coact.min().item():.4f}, {current_coact.max().item():.4f}]")
			axs[2].imshow(costs.cpu().numpy(), aspect='equal')
			axs[2].set_title(f"Costs\n[{costs.min().item():.4f}, {costs.max().item():.4f}]")
			fig.suptitle(f"Iteration {i} with cost {pair_cost:.4f}")
			plt.tight_layout()
			plt.show()

		i += 1


	plt.plot(merge_costs['max_considered_cost'], label='max considered cost')
	plt.plot(merge_costs['non_diag_costs_min'], label='non-diag costs min')
	plt.plot(merge_costs['non_diag_costs_max'], label='non-diag costs max')
	plt.xlabel("Iteration")
	plt.ylabel("Cost")
	plt.legend()