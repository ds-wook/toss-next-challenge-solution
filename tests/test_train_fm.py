import pytest
import torch
import sys
import os
from unittest.mock import patch, MagicMock
import numpy as np

# Add parent directory to path to import train_torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from train_fm import main
from utils.config import load_yaml


@pytest.fixture
def mock_data_loader(mock_toss_ad_data):
    """Create a mock FMDataLoader"""
    mock_loader = MagicMock()

    # Convert polars DataFrame to torch tensors for training data
    numerical_cols = [
        col
        for col in mock_toss_ad_data.columns
        if col not in ["gender", "age_group", "day_of_week", "hour", "clicked"]
    ]
    categorical_cols = ["gender", "age_group", "day_of_week", "hour"]

    # Create combined feature matrix (numerical + categorical)
    num_features = mock_toss_ad_data.select(numerical_cols).to_numpy()
    cat_features = np.random.randint(
        0, 10, size=(len(mock_toss_ad_data), len(categorical_cols))
    )
    seq = ["1,2,3,4,5,6,7,8,9,10"] * len(mock_toss_ad_data)

    # Combine all features into one matrix
    all_features = np.concatenate([num_features, cat_features], axis=1)
    targets = mock_toss_ad_data["clicked"].to_numpy().reshape(-1, 1)

    # Split into train/val (80/20)
    split_idx = int(0.8 * len(all_features))

    # Prepare train data
    train_x = torch.FloatTensor(all_features[:split_idx])
    train_y = torch.FloatTensor(targets[:split_idx])

    # Prepare validation data
    val_x = torch.FloatTensor(all_features[split_idx:])
    val_y = torch.FloatTensor(targets[split_idx:])

    # Test data (no targets)
    test_size = 100
    test_x = torch.FloatTensor(all_features[:test_size])
    test_ids = list(range(100))

    # Mock the load_train_data method to return 4 tensors (train_x, train_y, val_x, val_y)
    mock_loader.load_train_data.return_value = (
        train_x,
        train_y,
        seq[:split_idx],
        val_x,
        val_y,
        seq[split_idx:],
    )

    # Mock the load_test_data method to return test tensor and IDs
    mock_loader.load_test_data.return_value = (
        test_x,
        test_ids,
        seq[:test_size],
    )

    return mock_loader


def mock_load_yaml_side_effect(path):
    config = load_yaml(path)
    # Override only the specific attributes we want to test
    config.common.epochs = 1
    config.data.num_features = ["mock_feature"] * 112
    config.data.cat_features = ["mock_cat_feature"] * 5
    config.data.min_id_in_seq = 1
    config.data.max_id_in_seq = 10
    return config


@pytest.mark.parametrize(
    "model,use_seq_feature",
    [
        # models that use seq feature
        ("fm", True),
        ("deepfm", True),
        ("xdeepfm", True),
        ("ffm", True),
        ("dcn", True),
        ("dcn_v2", True),
        # models that do not use seq feature
        ("lr", False),
        ("fm", False),
        ("deepfm", False),
        ("xdeepfm", False),
        ("ffm", False),
        ("fibinet", False),
        ("dcn", False),
        ("dcn_v2", False),
    ],
)
def test_main_training_pipeline(mock_args, mock_data_loader, model, use_seq_feature):
    """Test the complete training pipeline"""
    # Mock the cfg attribute with categorical_field_dims
    mock_data_loader.categorical_field_dims = [10] * 5

    with (
        patch("train_fm.parse_args", return_value=mock_args),
        patch("train_fm.FMDataLoader", return_value=mock_data_loader),
        patch("train_fm.load_yaml", side_effect=mock_load_yaml_side_effect),
        patch("train_fm.validate_experiment_config"),
    ):
        mock_args.model = model
        mock_args.use_seq_feature = use_seq_feature

        # Run the main function
        main()

        # Verify data loader was called correctly
        mock_data_loader.load_train_data.assert_called_once_with(
            is_fm=True, is_test=False
        )
        mock_data_loader.load_test_data.assert_called_once_with(is_fm=True)
