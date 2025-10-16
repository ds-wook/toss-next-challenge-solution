import os
import sys
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

# Add parent directory to path to import train_boosting
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from train_boosting import _main


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing"""
    config = MagicMock()

    # Store configuration (features are now in store)
    config.store.cat_features = ["gender", "age_group", "hour"]
    config.store.num_features = [
        "l_feat_1",
        "l_feat_2",
        "l_feat_3",
        "feat_e_1",
        "feat_e_2",
        "feat_d_1",
        "feat_d_2",
        "feat_c_1",
        "feat_c_2",
        "feat_b_1",
        "feat_b_2",
        "feat_a_1",
        "feat_a_2",
        "history_a_1",
        "history_a_2",
        "history_b_1",
        "history_b_2",
    ]

    # Data configuration
    config.data.split_type = "stratified"
    config.data.n_splits = 5
    config.data.group = "inventory_id"  # Add group column for testing

    # Model configuration
    config.models.model_path = "res/models"
    config.models.results = "test_model"

    return config


@pytest.fixture
def mock_data_loader(mock_toss_ad_data):
    """Create a mock TreeDataLoader with subset of features"""
    mock_loader = MagicMock()

    # Use only the features defined in mock_config (hardcoded to avoid fixture call)
    cat_features = ["gender", "age_group", "hour"]
    num_features = [
        "l_feat_1",
        "l_feat_2",
        "l_feat_3",
        "feat_e_1",
        "feat_e_2",
        "feat_d_1",
        "feat_d_2",
        "feat_c_1",
        "feat_c_2",
        "feat_b_1",
        "feat_b_2",
        "feat_a_1",
        "feat_a_2",
        "history_a_1",
        "history_a_2",
        "history_b_1",
        "history_b_2",
    ]

    feature_cols = cat_features + num_features + ["clicked", "inventory_id"]
    subset_data = mock_toss_ad_data.select(feature_cols)

    # Split features and target
    train_x = subset_data.select(
        [col for col in subset_data.columns if col != "clicked"]
    )
    train_y = subset_data.select(["clicked"])

    mock_loader.load_train_data.return_value = (train_x, train_y)

    return mock_loader


def test_main_training_pipeline(mock_config, mock_data_loader):
    """Test the complete training pipeline for boosting models"""
    with (
        patch("train_boosting.TreeDataLoader", return_value=mock_data_loader),
        patch("train_boosting.instantiate") as mock_instantiate,
        patch("train_boosting.Path") as mock_path,
    ):
        # Create a real trainer mock
        mock_trainer = MagicMock()
        mock_trainer.run_cv_training.return_value = None
        mock_trainer.save_model.return_value = None
        mock_instantiate.return_value = mock_trainer

        # Mock Path behavior
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.__truediv__.return_value = "res/models/test_model.pkl"

        # Run the main function
        _main(mock_config)

        # Verify data loader was called correctly
        mock_data_loader.load_train_data.assert_called_once_with(is_boosting=True)

        # Verify trainer was instantiated with correct parameters
        expected_features = (
            mock_config.store.num_features + mock_config.store.cat_features
        )
        # Get the actual call to check logger separately
        assert mock_instantiate.call_count == 1
        call_args = mock_instantiate.call_args
        assert call_args[0][0] == mock_config.models
        assert call_args[1]["features"] == expected_features
        assert call_args[1]["cat_features"] == mock_config.store.cat_features
        assert call_args[1]["n_splits"] == mock_config.data.n_splits
        assert call_args[1]["split_type"] == mock_config.data.split_type
        assert "logger" in call_args[1]  # Verify logger is passed

        # Verify training was called
        mock_trainer.run_cv_training.assert_called_once()

        # Verify model was saved
        mock_trainer.save_model.assert_called_once_with("res/models/test_model.pkl")


def test_data_conversion_to_pandas(mock_config, mock_data_loader):
    """Test that polars DataFrames are correctly converted to pandas"""
    with (
        patch("train_boosting.TreeDataLoader", return_value=mock_data_loader),
        patch("train_boosting.instantiate") as mock_instantiate,
        patch("train_boosting.Path") as mock_path,
    ):
        # Create a trainer that captures the data passed to it
        captured_data = {}

        def capture_training_data(train_x, train_y, groups=None):
            captured_data["train_x"] = train_x
            captured_data["train_y"] = train_y
            captured_data["groups"] = groups

        mock_trainer = MagicMock()
        mock_trainer.run_cv_training.side_effect = capture_training_data
        mock_trainer.save_model.return_value = None
        mock_instantiate.return_value = mock_trainer

        # Mock Path behavior
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.__truediv__.return_value = "res/models/test_model.pkl"

        # Run the main function
        _main(mock_config)

        # Verify that the data is polars DataFrame
        assert isinstance(captured_data["train_x"], pl.DataFrame), (
            "train_x should be a polars DataFrame"
        )
        assert isinstance(captured_data["train_y"], pl.DataFrame), (
            "train_y should be a polars DataFrame"
        )

        # Verify data shapes
        expected_features = (
            mock_config.store.num_features + mock_config.store.cat_features
        )
        # train_x includes all features except 'clicked' (includes inventory_id for grouping)
        assert captured_data["train_x"].shape[1] == len(expected_features) + 1, (
            "Feature count mismatch - should include group column"
        )
        assert captured_data["train_y"].shape[1] == 1, "Target should have 1 column"
