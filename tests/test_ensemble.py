import os
import sys
import unittest

import numpy as np

# Add parent directory to path to import ensemble
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from ensemble import ensemble_predictions


class TestEnsemblePredictions(unittest.TestCase):
    """Test cases for ensemble_predictions function."""

    def setUp(self):
        """Set up test data before each test method."""
        # Create sample predictions
        self.pred1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.pred2 = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        self.pred3 = np.array([0.15, 0.25, 0.35, 0.45, 0.55])

        self.predictions = [self.pred1, self.pred2, self.pred3]
        self.weights = [0.4, 0.3, 0.3]

        # Expected linear ensemble result (manual calculation)
        self.expected_linear = 0.4 * self.pred1 + 0.3 * self.pred2 + 0.3 * self.pred3

    def test_linear_ensemble(self):
        """Test linear ensemble method."""
        result = ensemble_predictions(self.predictions, self.weights, "linear")

        np.testing.assert_array_almost_equal(result, self.expected_linear)
        assert result.shape == self.pred1.shape

    def test_linear_ensemble_equal_weights(self):
        """Test linear ensemble with equal weights (default)."""
        result = ensemble_predictions(self.predictions, method="linear")
        expected = np.mean(self.predictions, axis=0)

        np.testing.assert_array_almost_equal(result, expected)

    def test_harmonic_ensemble(self):
        """Test harmonic ensemble method."""
        result = ensemble_predictions(self.predictions, self.weights, "harmonic")

        # Manual calculation of harmonic mean
        harmonic_preds = [1 / p for p in self.predictions]
        weighted_harmonic = np.average(harmonic_preds, weights=self.weights, axis=0)
        expected = 1 / weighted_harmonic

        np.testing.assert_array_almost_equal(result, expected)

    def test_geometric_ensemble(self):
        """Test geometric ensemble method."""
        result = ensemble_predictions(self.predictions, self.weights, "geometric")

        # Manual calculation of geometric mean
        log_preds = [np.log(p) for p in self.predictions]
        weighted_log_mean = np.average(log_preds, weights=self.weights, axis=0)
        expected = np.exp(weighted_log_mean)

        np.testing.assert_array_almost_equal(result, expected)

    def test_rank_ensemble(self):
        """Test rank ensemble method."""
        result = ensemble_predictions(self.predictions, self.weights, "rank")

        # Check that result is normalized to [0, 1] range
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        assert result.shape == self.pred1.shape

    def test_sigmoid_ensemble(self):
        """Test sigmoid ensemble method."""
        # Use predictions in (0, 1) range for sigmoid
        sigmoid_preds = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.2, 0.3, 0.4]),
            np.array([0.15, 0.25, 0.35]),
        ]

        result = ensemble_predictions(sigmoid_preds, self.weights, "sigmoid")

        # Check that result is in (0, 1) range
        assert np.all(result > 0)
        assert np.all(result < 1)
        assert result.shape == sigmoid_preds[0].shape

    def test_single_prediction(self):
        """Test ensemble with single prediction."""
        single_pred = [self.pred1]
        single_weight = [1.0]

        result = ensemble_predictions(single_pred, single_weight, "linear")
        np.testing.assert_array_equal(result, self.pred1)

    def test_two_predictions(self):
        """Test ensemble with two predictions."""
        two_preds = [self.pred1, self.pred2]
        two_weights = [0.6, 0.4]

        result = ensemble_predictions(two_preds, two_weights, "linear")
        expected = 0.6 * self.pred1 + 0.4 * self.pred2

        np.testing.assert_array_almost_equal(result, expected)

    def test_different_array_sizes(self):
        """Test ensemble with predictions of different sizes."""
        pred_small = np.array([0.1, 0.2])
        pred_large = np.array([0.1, 0.2, 0.3, 0.4])

        with self.assertRaises(ValueError):
            ensemble_predictions([pred_small, pred_large], [0.5, 0.5], "linear")

    def test_weights_sum_not_one(self):
        """Test that weights must sum to 1.0."""
        invalid_weights = [0.5, 0.3, 0.1]  # Sum = 0.9

        with self.assertRaises(AssertionError):
            ensemble_predictions(self.predictions, invalid_weights, "linear")

    def test_weights_length_mismatch(self):
        """Test that weights length must match predictions length."""
        invalid_weights = [0.5, 0.5]  # Length = 2, but predictions length = 3

        with self.assertRaises(ValueError):
            ensemble_predictions(self.predictions, invalid_weights, "linear")

    def test_unknown_method(self):
        """Test that unknown method raises ValueError."""
        with self.assertRaises(ValueError):
            ensemble_predictions(self.predictions, self.weights, "unknown_method")

    def test_empty_predictions_list(self):
        """Test that empty predictions list raises error."""
        with self.assertRaises(AssertionError):
            ensemble_predictions([], [], "linear")

    def test_none_weights_defaults_to_equal(self):
        """Test that None weights defaults to equal weights."""
        result = ensemble_predictions(self.predictions, None, "linear")
        expected = np.mean(self.predictions, axis=0)

        np.testing.assert_array_almost_equal(result, expected)

    def test_negative_predictions_harmonic(self):
        """Test harmonic ensemble with negative predictions."""
        negative_preds = [
            np.array([-0.1, 0.2, 0.3]),
            np.array([0.1, -0.2, 0.3]),
            np.array([0.1, 0.2, -0.3]),
        ]

        # Should handle negative values gracefully
        result = ensemble_predictions(negative_preds, self.weights, "harmonic")
        assert result.shape == negative_preds[0].shape

    def test_zero_predictions_geometric(self):
        """Test geometric ensemble with zero predictions."""
        zero_preds = [
            np.array([0.0, 0.2, 0.3]),
            np.array([0.1, 0.0, 0.3]),
            np.array([0.1, 0.2, 0.0]),
        ]

        # Should handle zero values gracefully
        result = ensemble_predictions(zero_preds, self.weights, "geometric")
        assert result.shape == zero_preds[0].shape

    def test_boundary_values_sigmoid(self):
        """Test sigmoid ensemble with boundary values (0 and 1)."""
        # Use safer boundary values that avoid exact 0 and 1
        boundary_preds = [
            np.array([0.01, 0.5, 0.99]),
            np.array([0.1, 0.5, 0.9]),
            np.array([0.05, 0.5, 0.95]),
        ]

        result = ensemble_predictions(boundary_preds, self.weights, "sigmoid")

        # Result should be in (0, 1) range
        assert np.all(result > 0)
        assert np.all(result < 1)
        assert result.shape == boundary_preds[0].shape

    def test_large_arrays(self):
        """Test ensemble with large arrays."""
        large_pred1 = np.random.random(1000)
        large_pred2 = np.random.random(1000)
        large_pred3 = np.random.random(1000)

        large_predictions = [large_pred1, large_pred2, large_pred3]

        result = ensemble_predictions(large_predictions, self.weights, "linear")

        assert result.shape == (1000,)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_all_methods_consistency(self):
        """Test that all methods return arrays of the same shape."""
        methods = ["linear", "harmonic", "geometric", "rank", "sigmoid"]
        results = {}

        for method in methods:
            results[method] = ensemble_predictions(
                self.predictions, self.weights, method
            )

        # All results should have the same shape
        shapes = [result.shape for result in results.values()]
        assert all(shape == shapes[0] for shape in shapes)

        # All results should be finite
        for method, result in results.items():
            assert not np.any(np.isnan(result)), f"NaN found in {method} result"
            assert not np.any(np.isinf(result)), f"Inf found in {method} result"

    def test_weights_precision(self):
        """Test that weights with small precision errors are handled correctly."""
        # Weights that sum to approximately 1.0 but not exactly
        precise_weights = [0.3333333333333333, 0.3333333333333333, 0.3333333333333334]

        result = ensemble_predictions(self.predictions, precise_weights, "linear")
        assert result.shape == self.pred1.shape


if __name__ == "__main__":
    unittest.main()
