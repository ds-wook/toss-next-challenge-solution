import pytest
import numpy as np
import polars as pl
from unittest.mock import MagicMock


@pytest.fixture
def mock_toss_ad_data():
    np.random.seed(42)
    n_rows = 1000

    # Numerical features (scaled 0-1)
    num_features = [
        "l_feat_1",
        "l_feat_2",
        "l_feat_3",
        "l_feat_4",
        "l_feat_5",
        "l_feat_6",
        "l_feat_7",
        "l_feat_8",
        "l_feat_9",
        "l_feat_10",
        "l_feat_11",
        "l_feat_12",
        "l_feat_13",
        "l_feat_14",
        "l_feat_15",
        "l_feat_16",
        "l_feat_17",
        "l_feat_18",
        "l_feat_19",
        "l_feat_20",
        "l_feat_21",
        "l_feat_22",
        "l_feat_23",
        "l_feat_24",
        "l_feat_25",
        "l_feat_26",
        "l_feat_27",
        "feat_e_1",
        "feat_e_2",
        "feat_e_3",
        "feat_e_4",
        "feat_e_5",
        "feat_e_6",
        "feat_e_7",
        "feat_e_8",
        "feat_e_9",
        "feat_e_10",
        "feat_d_1",
        "feat_d_2",
        "feat_d_3",
        "feat_d_4",
        "feat_d_5",
        "feat_d_6",
        "feat_c_1",
        "feat_c_2",
        "feat_c_3",
        "feat_c_4",
        "feat_c_5",
        "feat_c_6",
        "feat_c_7",
        "feat_c_8",
        "feat_b_1",
        "feat_b_2",
        "feat_b_3",
        "feat_b_4",
        "feat_b_5",
        "feat_b_6",
        "feat_a_1",
        "feat_a_2",
        "feat_a_3",
        "feat_a_4",
        "feat_a_5",
        "feat_a_6",
        "feat_a_7",
        "feat_a_8",
        "feat_a_9",
        "feat_a_10",
        "feat_a_11",
        "feat_a_12",
        "feat_a_13",
        "feat_a_14",
        "feat_a_15",
        "feat_a_16",
        "feat_a_17",
        "feat_a_18",
        "history_a_1",
        "history_a_2",
        "history_a_3",
        "history_a_4",
        "history_a_5",
        "history_a_6",
        "history_a_7",
        "history_b_1",
        "history_b_2",
        "history_b_3",
        "history_b_4",
        "history_b_5",
        "history_b_6",
        "history_b_7",
        "history_b_8",
        "history_b_9",
        "history_b_10",
        "history_b_11",
        "history_b_12",
        "history_b_13",
        "history_b_14",
        "history_b_15",
        "history_b_16",
        "history_b_17",
        "history_b_18",
        "history_b_19",
        "history_b_20",
        "history_b_21",
        "history_b_22",
        "history_b_23",
        "history_b_24",
        "history_b_25",
        "history_b_26",
        "history_b_27",
        "history_b_28",
        "history_b_29",
        "history_b_30",
    ]

    # Create data dictionary
    data = {}

    # Add numerical features (scaled 0-1)
    for feature in num_features:
        data[feature] = np.random.uniform(0, 1, n_rows)

    # Add categorical features
    data["gender"] = np.random.choice(["M", "F"], n_rows)
    data["age_group"] = np.random.choice(
        ["10-19", "20-29", "30-39", "40-49", "50-59", "60+"], n_rows
    )
    data["day_of_week"] = np.random.choice([0, 1, 2, 3, 4, 5, 6], n_rows)
    data["hour"] = np.random.choice(range(24), n_rows)
    data["inventory_id"] = np.random.choice(range(24), n_rows)

    # Add target with 50/50 ratio
    data["clicked"] = np.random.choice([0, 1], n_rows, p=[0.5, 0.5])

    return pl.DataFrame(data)


@pytest.fixture
def mock_args():
    """Create mock args for testing"""
    args = MagicMock()
    args.model = "lr"
    args.batch_size = 8
    args.is_test = False
    args.postfix = "pytest"
    args.use_seq_feature = False
    return args
