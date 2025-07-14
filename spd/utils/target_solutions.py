"""Target patterns for evaluating causal importance matrices.

This module provides abstractions for testing whether learned sparsity patterns
match expected target solutions in toy models:

- TargetPattern classes define expected sparsity patterns (Identity, DenseColumns)
- TargetSolution maps model components to their expected patterns
- Evaluation uses a discrete distance metric that counts elements deviating beyond
  a tolerance threshold, making it robust to small values from inactive components
"""

from abc import ABC, abstractmethod
from typing import Literal, override

import torch
from jaxtyping import Float, Int
from torch import Tensor

from .linear_sum_assignment import linear_sum_assignment


def permute_to_identity_greedy(
    ci_vals: Float[Tensor, "batch C"],
) -> tuple[Float[Tensor, "batch C"], Int[Tensor, " C"]]:
    """Permute matrix to make it as close to identity as possible using greedy algorithm.

    Returns:
        - Permuted mask
        - Permutation indices
    """
    if ci_vals.ndim != 2:
        raise ValueError(f"Mask must have 2 dimensions, got {ci_vals.ndim}")

    batch, C = ci_vals.shape
    effective_rows = min(batch, C)

    perm = []
    used = set()
    for i in range(effective_rows):
        sorted_indices = torch.argsort(ci_vals[i, :], descending=True)
        chosen = next(
            (col.item() for col in sorted_indices if col.item() not in used),
            sorted_indices[0].item(),
        )
        perm.append(chosen)
        used.add(chosen)

    # Add remaining columns
    remaining = sorted(set(range(C)) - used)
    perm.extend(remaining)

    perm_indices = torch.tensor(perm, device=ci_vals.device, dtype=torch.long)
    return ci_vals[:, perm_indices], perm_indices


def permute_to_identity_hungarian(
    ci_vals: Float[Tensor, "batch C"],
) -> tuple[Float[Tensor, "batch C"], Int[Tensor, " C"]]:
    """Permute matrix to make it as close to identity as possible using Hungarian algorithm.

    Returns:
        - Permuted mask
        - Permutation indices
    """
    if ci_vals.ndim != 2:
        raise ValueError(f"Mask must have 2 dimensions, got {ci_vals.ndim}")

    batch, C = ci_vals.shape
    device = ci_vals.device
    effective_rows = min(batch, C)

    # Hungarian algorithm on the effective_rows x C submatrix
    cost_matrix = -ci_vals[:effective_rows].detach().cpu().numpy()
    _, col_indices = linear_sum_assignment(cost_matrix)

    # Build complete permutation
    assigned_cols = set(col_indices.tolist())
    unassigned_cols = sorted(set(range(C)) - assigned_cols)

    perm_list = list(col_indices) + unassigned_cols
    perm_indices = torch.tensor(perm_list, device=device, dtype=torch.long)

    return ci_vals[:, perm_indices], perm_indices


def permute_to_dense(
    ci_vals: Float[Tensor, "batch C"],
) -> tuple[Float[Tensor, "batch C"], Int[Tensor, " C"]]:
    """Permute columns by density, placing highest mass columns first.

    Args:
        ci_vals: The causal importance values matrix

    Returns:
        - Permuted matrix with densest columns first
        - Permutation indices
    """
    if ci_vals.ndim != 2:
        raise ValueError(f"Matrix must have 2 dimensions, got {ci_vals.ndim}")

    # Sort columns by total mass in descending order
    column_sums = ci_vals.sum(dim=0)
    perm_indices = torch.argsort(column_sums, descending=True)

    return ci_vals[:, perm_indices], perm_indices


class TargetPattern(ABC):
    """Base class for target patterns."""

    def _verify_inputs(self, ci_array: torch.Tensor) -> None:
        """Verify that input is a 2D torch tensor."""
        if ci_array.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {ci_array.shape}")

    @abstractmethod
    def distance_from(self, ci_array: torch.Tensor, tolerance: float = 0.1) -> int:
        """Discrete distance: count of elements deviating from expected pattern.

        Uses a tolerance threshold to avoid sensitivity to small values from
        inactive components. Elements are counted as "off" if they deviate
        from the expected value by more than the tolerance.
        """
        pass

    @abstractmethod
    def permute_for_display(self, ci_array: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Permute the causal importance array for optimal display of this pattern.

        Args:
            ci_array: The causal importance array to permute

        Returns:
            - Permuted array
            - Permutation indices
        """
        pass


class IdentityPattern(TargetPattern):
    """Identity pattern: expects one-to-one feature to component mapping.

    Each feature should activate exactly one component (up to permutation).
    Counts elements that violate this pattern beyond the tolerance threshold.
    """

    def __init__(
        self,
        n_features: int,
        apply_permutation: bool = True,
        method: Literal["hungarian", "greedy"] = "hungarian",
    ):
        self.n_features = n_features
        self.apply_permutation = apply_permutation
        self.method = method

    @override
    def _verify_inputs(self, ci_array: torch.Tensor) -> None:
        super()._verify_inputs(ci_array)
        n, c = ci_array.shape
        if n != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {n}")
        if c < self.n_features:
            raise ValueError(f"Expected at least {self.n_features} components, got {c}")

    @override
    def distance_from(self, ci_array: torch.Tensor, tolerance: float = 0.1) -> int:
        self._verify_inputs(ci_array)
        if self.apply_permutation:
            if self.method == "hungarian":
                ci_array = permute_to_identity_hungarian(ci_array)[0]
            else:
                ci_array = permute_to_identity_greedy(ci_array)[0]
        n, c = ci_array.shape
        size = min(n, c)

        # Off-diagonal errors + on-diagonal errors
        mask = torch.ones_like(ci_array, dtype=torch.bool)
        mask[:size, :size].fill_diagonal_(False)
        off_diag_errors = torch.sum(ci_array[mask] > tolerance)
        on_diag_errors = torch.sum(torch.diag(ci_array[:size, :size]) < (1 - tolerance))
        return int(off_diag_errors + on_diag_errors)

    @override
    def permute_for_display(self, ci_array: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Permute to show identity pattern."""
        if self.method == "hungarian":
            return permute_to_identity_hungarian(ci_array)
        else:
            return permute_to_identity_greedy(ci_array)


class DenseColumnsPattern(TargetPattern):
    """Dense columns pattern: at most K components should be active.

    Expects sparsity where only K columns (components) have non-zero entries.
    Counts the number of entries in excess columns that exceed the tolerance.
    """

    def __init__(self, k: int):
        self.k = k

    @override
    def _verify_inputs(self, ci_array: torch.Tensor) -> None:
        super()._verify_inputs(ci_array)
        _, c = ci_array.shape
        if c < self.k:
            raise ValueError(f"Expected at least {self.k} columns, got {c}")

    @override
    def distance_from(self, ci_array: torch.Tensor, tolerance: float = 0.1) -> int:
        self._verify_inputs(ci_array)
        column_sums = (torch.clamp(ci_array, 0, 1) > tolerance).sum(dim=0)
        # torch.kthvalue returns the kth smallest, so we need to sort descending first
        sorted_sums, _ = torch.sort(column_sums, descending=True)
        return int(sorted_sums[self.k :].sum())

    @override
    def permute_for_display(self, ci_array: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Permute to show dense columns pattern."""
        return permute_to_dense(ci_array)


class TargetSolution:
    """Collection of expected patterns for different modules in a model."""

    def __init__(self, module_targets: dict[str, TargetPattern]):
        self.module_targets = module_targets

    def distance_from(self, ci_arrays: dict[str, torch.Tensor], tolerance: float = 0.1) -> int:
        """Total number of elements that are off across all modules."""
        target_keys = set(self.module_targets.keys())
        ci_keys = set(ci_arrays.keys())

        if target_keys != ci_keys:
            missing = target_keys - ci_keys
            extra = ci_keys - target_keys
            raise ValueError(f"Keys mismatch. Missing: {missing}, Extra: {extra}")

        return sum(
            self.module_targets[name].distance_from(arr, tolerance)
            for name, arr in ci_arrays.items()
        )

    def permute_to_target(
        self, ci_vals: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Permute causal importance matrices to best display their target patterns.

        Args:
            ci_vals: Dictionary of causal importance matrices by module name

        Returns:
            - Dictionary of permuted matrices
            - Dictionary of permutation indices
        """
        permuted_ci = {}
        perm_indices = {}

        for module_name, ci_matrix in ci_vals.items():
            if module_name in self.module_targets:
                pattern = self.module_targets[module_name]
                permuted_ci[module_name], perm_indices[module_name] = pattern.permute_for_display(
                    ci_matrix
                )
            else:
                # Default for modules not in target
                permuted_ci[module_name], perm_indices[module_name] = permute_to_identity_greedy(
                    ci_matrix
                )

        return permuted_ci, perm_indices


def compute_target_metrics(
    causal_importances: dict[str, torch.Tensor],
    target_solution: TargetSolution,
    tolerance: float = 0.1,
) -> dict[str, float]:
    """Compute target solution distance metrics.

    Args:
        causal_importances: Dictionary of causal importance tensors
        target_solution: The target solution to compare against
        tolerance: Tolerance for pattern matching

    Returns:
        Dictionary of target distance metrics
    """
    metrics = {}

    # Total error across all modules
    metrics["target_solution_error/total"] = target_solution.distance_from(
        causal_importances, tolerance
    )
    metrics["target_solution_error/total_0p2"] = target_solution.distance_from(
        causal_importances, 0.2
    )

    # Per-module errors
    for module_name, pattern in target_solution.module_targets.items():
        module_error = pattern.distance_from(causal_importances[module_name], tolerance)
        metrics[f"target_solution_error/{module_name}"] = module_error

    return metrics
