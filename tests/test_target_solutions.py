import torch

from spd.utils.target_solutions import DenseColumnsPattern, IdentityPattern, TargetSolution


class TestIdentityPattern:
    def test_perfect_identity_distance_zero(self):
        """Perfect identity matrix should have distance 0."""
        pattern = IdentityPattern(n_features=3)
        ci_array = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )
        assert pattern.distance_from(ci_array, tolerance=0.1) == 0

    def test_within_tolerance_identity(self):
        """Single off-diagonal element above tolerance."""
        pattern = IdentityPattern(n_features=3)
        ci_array = torch.tensor(
            [
                [0.95, 0.01, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.05],
                [0.0, 0.0, 0.99, 0.0],
            ]
        )
        assert pattern.distance_from(ci_array, tolerance=0.1) == 0

    def test_one_off_diagonal_error(self):
        """Single off-diagonal element above tolerance."""
        pattern = IdentityPattern(n_features=3)
        ci_array = torch.tensor(
            [
                [1.0, 0.2, 0.0, 0.0],  # 0.2 > 0.1 tolerance
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        assert pattern.distance_from(ci_array, tolerance=0.1) == 1

    def test_multiple_errors(self):
        """Multiple off-diagonal and diagonal errors."""
        pattern = IdentityPattern(n_features=3)
        ci_array = torch.tensor(
            [
                [0.7, 0.3, 0.0, 0.2],
                [0.2, 0.95, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],  # perfect row
            ]
        )
        assert pattern.distance_from(ci_array, tolerance=0.1) == 4


class TestDenseColumnsPattern:
    def test_exactly_k_columns_distance_zero(self):
        """When exactly k columns are active, distance is 0."""
        pattern = DenseColumnsPattern(k=2)
        ci_array = torch.tensor(
            [
                [0.5, 0.0, 0.3, 0.0],
                [0.6, 0.0, 0.4, 0.0],
                [0.7, 0.0, 0.5, 0.0],
            ]
        )
        # Columns 0 and 2 active, columns 1 and 3 inactive
        assert pattern.distance_from(ci_array, tolerance=0.1) == 0

    def test_one_excess_column(self):
        """One column beyond k has entries."""
        pattern = DenseColumnsPattern(k=1)
        ci_array = torch.tensor(
            [
                [0.5, 0.2, 0.0, 0.0],
                [0.6, 0.3, 0.0, 0.0],
            ]
        )
        # Columns 0 and 1 active, but k=1, so column 1 has 2 excess entries
        assert pattern.distance_from(ci_array, tolerance=0.1) == 2

    def test_multiple_excess_columns(self):
        """Multiple columns beyond k have entries."""
        pattern = DenseColumnsPattern(k=2)
        ci_array = torch.tensor(
            [
                [0.5, 0.4, 0.3, 0.2, 0.1],
                [0.6, 0.5, 0.4, 0.3, 0.2],
                [0.7, 0.6, 0.5, 0.4, 0.3],
            ]
        )
        # All 5 columns active, but k=2
        # Column 4 has values [0.1, 0.2, 0.3], only 2 are > 0.1
        # So excess entries: col 2 (3) + col 3 (3) + col 4 (2) = 8
        assert pattern.distance_from(ci_array, tolerance=0.1) == 8


class TestTargetSolution:
    def test_combined_errors_from_modules(self):
        """Errors from multiple modules should sum."""
        solution = TargetSolution(
            {"module1": IdentityPattern(n_features=2), "module2": DenseColumnsPattern(k=1)}
        )
        ci_arrays = {
            "module1": torch.tensor(
                [
                    [0.8, 0.2, 0.0],  # diagonal low, off-diag high
                    [0.0, 1.0, 0.0],
                ]
            ),
            "module2": torch.tensor(
                [
                    [0.5, 0.3, 0.0, 0.0],  # 2 columns active but k=1
                    [0.0, 0.4, 0.0, 0.0],
                ]
            ),
        }
        # module1: 1 diagonal + 1 off-diagonal = 2 errors
        # module2: column 0 has 1 entry (0.5), column 1 has 1 entry (0.4)
        # With k=1, we keep column 0, so column 1's 1 entry is excess
        # Total: 2 + 1 = 3 errors
        assert solution.distance_from(ci_arrays, tolerance=0.1) == 3
        # when we increase the tolerance, the distance should now be only 1
        assert solution.distance_from(ci_arrays, tolerance=0.2) == 1

    def test_permute_to_target(self):
        """Test that TargetSolution can permute CI arrays to match patterns."""
        solution = TargetSolution(
            {
                "identity_module": IdentityPattern(n_features=2),
                "dense_module": DenseColumnsPattern(k=1),
            }
        )

        # Create CI arrays that need permutation
        ci_arrays = {
            "identity_module": torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.2, 0.0]]),
            "dense_module": torch.tensor([[0.3, 0.0, 0.8], [0.2, 0.0, 0.9], [0.1, 0.0, 0.7]]),
        }

        # Expected results after permutation
        expected_identity = torch.tensor([[0.9, 0.1, 0.0], [0.2, 0.8, 0.0]])

        expected_dense = torch.tensor([[0.8, 0.3, 0.0], [0.9, 0.2, 0.0], [0.7, 0.1, 0.0]])

        # Permute to match target patterns
        permuted_ci, perm_indices = solution.permute_to_target(ci_arrays)

        # Check results match expected
        assert torch.allclose(permuted_ci["identity_module"], expected_identity)
        assert torch.allclose(permuted_ci["dense_module"], expected_dense)

        # Check indices are correct
        assert torch.equal(perm_indices["identity_module"], torch.tensor([1, 0, 2]))
        assert torch.equal(perm_indices["dense_module"], torch.tensor([2, 0, 1]))
