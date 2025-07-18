"""Target patterns for evaluating causal importance matrices.

This module provides abstractions for testing whether learned sparsity patterns
match expected target solutions in toy models:

- TargetCIPattern classes define expected sparsity patterns (Identity, DenseColumns)
- TargetCISolution maps model components to their expected patterns
- Evaluation uses a discrete distance metric that counts elements deviating beyond
  a tolerance threshold, making it robust to small values from inactive components
"""

import fnmatch
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


class TargetCIPattern(ABC):
    """Base class for target patterns."""

    def _verify_inputs(self, ci_array: Float[Tensor, "batch C"]) -> None:
        """Verify that input is a 2D torch tensor."""
        if ci_array.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {ci_array.shape}")

    @abstractmethod
    def distance_from(self, ci_array: Float[Tensor, "batch C"], tolerance: float = 0.1) -> int:
        """Discrete distance: count of elements deviating from expected pattern.

        Uses a tolerance threshold to avoid sensitivity to small values from
        inactive components. Elements are counted as "off" if they deviate
        from the expected value by more than the tolerance.
        """
        pass

    @abstractmethod
    def permute_for_display(self, ci_array: Float[Tensor, "batch C"]) -> tuple[Float[Tensor, "batch C"], Int[Tensor, " C"]]:
        """Permute the causal importance array for optimal display of this pattern.

        Args:
            ci_array: The causal importance array to permute

        Returns:
            - Permuted array
            - Permutation indices
        """
        pass


class IdentityCIPattern(TargetCIPattern):
    """Identity pattern: expects one-to-one feature to component mapping.

    Each feature should activate exactly one component (up to permutation).
    Counts elements that violate this pattern beyond the tolerance threshold.
    """

    def __init__(
        self,
        n_features: int,
        apply_permutation: bool = True,
        method: Literal["hungarian", "greedy", "auto"] = "auto",
    ):
        self.n_features = n_features
        self.apply_permutation = apply_permutation
        self.method = method

    @override
    def _verify_inputs(self, ci_array: Float[Tensor, "batch C"]) -> None:
        super()._verify_inputs(ci_array)
        n, c = ci_array.shape
        if n != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {n}")
        if c < self.n_features:
            raise ValueError(f"Expected at least {self.n_features} components, got {c}")

    @override
    def distance_from(self, ci_array: Float[Tensor, "batch C"], tolerance: float = 0.1) -> int:
        self._verify_inputs(ci_array)
        if self.apply_permutation:
            if self.method == "hungarian" or (self.method == "auto" and min(ci_array.shape) < 500):
                ci_array = permute_to_identity_hungarian(ci_array)[0]
            else:
                ci_array = permute_to_identity_greedy(ci_array)[0]

        size = min(ci_array.shape)
        # Off-diagonal errors + on-diagonal errors
        mask = torch.ones_like(ci_array, dtype=torch.bool)
        mask[:size, :size].fill_diagonal_(False)
        off_diag_errors = torch.sum(ci_array[mask] > tolerance)
        on_diag_errors = torch.sum(torch.diag(ci_array[:size, :size]) < (1 - tolerance))
        return int(off_diag_errors + on_diag_errors)

    @override
    def permute_for_display(self, ci_array: Float[Tensor, "batch C"]) -> tuple[Float[Tensor, "batch C"], Int[Tensor, " C"]]:
        """Permute to show identity pattern."""
        if self.method == "hungarian" or (self.method == "auto" and min(ci_array.shape) < 500):
            return permute_to_identity_hungarian(ci_array)
        else:
            return permute_to_identity_greedy(ci_array)


class DenseCIPattern(TargetCIPattern):
    """Dense columns pattern: exactly K components should be active.

    Expects sparsity where exactly K columns (components) have non-zero entries.
    
    Args:
        k: Number of columns that should be active
        min_entries: Minimum number of strong activations (> 1-tolerance) 
                    required for a column to be considered "active"
    
    Error computation:
        - For first k columns: count how many are missing required strong activations
        - For remaining columns: count any activations (should be completely inactive)
        - Total error = active_column_deficits + inactive_column_violations
    """

    def __init__(self, k: int, min_entries: int = 1):
        self.k = k
        self.min_entries = min_entries

    @override
    def _verify_inputs(self, ci_array: Float[Tensor, "batch C"]) -> None:
        super()._verify_inputs(ci_array)
        _, c = ci_array.shape
        if c < self.k:
            raise ValueError(f"Expected at least {self.k} columns, got {c}")

    @override
    def distance_from(self, ci_array: Float[Tensor, "batch C"], tolerance: float = 0.1) -> int:
        self._verify_inputs(ci_array)
        sorted_ci = permute_to_dense(ci_array)[0]
        
        strong_activations_per_column = (sorted_ci >= 1 - tolerance).sum(dim=0)
        missing_strong_activations = torch.clamp(self.min_entries - strong_activations_per_column, min=0)
        first_k_column_error = missing_strong_activations[:self.k].sum().item()
        
        weak_activations_per_column = (sorted_ci > tolerance).sum(dim=0)
        inactive_column_error = weak_activations_per_column[self.k:].sum().item()
        
        return int(first_k_column_error + inactive_column_error)

    @override
    def permute_for_display(self, ci_array: Float[Tensor, "batch C"]) -> tuple[Float[Tensor, "batch C"], Int[Tensor, " C"]]:
        """Permute to show dense columns pattern."""
        return permute_to_dense(ci_array)


class TargetCISolution:
    """Collection of expected patterns for different modules in a model.

    The module_targets dictionary can use fnmatch-style patterns as keys:
    - Explicit module names: {"layers.0.mlp_in": IdentityCIPattern(...)}
    - Wildcard patterns: {"layers.*.mlp_in": IdentityCIPattern(...)}
    - Mixed patterns: {"layers.*.mlp_*": pattern, "specific.module": pattern}

    Patterns are expanded at runtime when given actual module names.
    First matching pattern wins for each module name.
    """

    def __init__(
        self, module_targets: dict[str, TargetCIPattern], expected_matches: int | None = None
    ):
        """Initialize target solution with pattern mappings.

        Args:
            module_targets: Dictionary mapping module name patterns to target patterns.
                Keys can be exact module names or fnmatch-style patterns (e.g., "layers.*.mlp_in").
            expected_matches: Optional validation - expected total number of modules that
                should match the patterns. If provided, expand_module_targets will validate
                that exactly this many modules match.
        """
        self.module_targets = module_targets
        self.expected_matches = expected_matches

    def expand_module_targets(
        self, module_names: list[str], validate: bool = True
    ) -> dict[str, TargetCIPattern]:
        """Expand patterns to concrete module name -> TargetCIPattern mappings."""
        result = {}
        for name in module_names:
            for pattern, target in self.module_targets.items():
                if fnmatch.fnmatch(name, pattern):
                    result[name] = target
                    break

        if validate and self.expected_matches and len(result) != self.expected_matches:
            raise ValueError(
                f"Expected {self.expected_matches} matches, got {len(result)}: {sorted(result.keys())}"
            )

        return result

    def distance_from(self, ci_arrays: dict[str, Float[Tensor, "batch C"]], tolerance: float = 0.1) -> int:
        """Total number of elements that are off across all modules."""
        expanded_targets = self.expand_module_targets(list(ci_arrays.keys()))

        return sum(
            target.distance_from(ci_arrays[name], tolerance)
            for name, target in expanded_targets.items()
        )

    def permute_to_target(
        self, ci_vals: dict[str, Float[Tensor, "batch C"]]
    ) -> tuple[dict[str, Float[Tensor, "batch C"]], dict[str, Int[Tensor, " C"]]]:
        """Permute causal importance matrices to best display their target patterns.

        Args:
            ci_vals: Dictionary of causal importance matrices by module name

        Returns:
            - Dictionary of permuted matrices
            - Dictionary of permutation indices
        """
        expanded_targets = self.expand_module_targets(list(ci_vals.keys()))
        permuted_ci = {}
        perm_indices = {}

        for module_name, ci_matrix in ci_vals.items():
            if module_name in expanded_targets:
                pattern = expanded_targets[module_name]
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
    causal_importances: dict[str, Float[Tensor, "batch C"]],
    target_solution: TargetCISolution,
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
    expanded_targets = target_solution.expand_module_targets(list(causal_importances.keys()))
    for module_name, pattern in expanded_targets.items():
        module_error = pattern.distance_from(causal_importances[module_name], tolerance)
        metrics[f"target_solution_error/{module_name}"] = module_error

    return metrics
