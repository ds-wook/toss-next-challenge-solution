"""
Configuration utilities
"""

import argparse
from dataclasses import dataclass
from typing import List

import yaml
from easydict import EasyDict


@dataclass
class ModelConfig:
    """Model configuration"""

    batch_size: int = 4096
    epochs: int = 10
    learning_rate: float = 1e-3
    seed: int = 42
    lstm_hidden: int = 64
    hidden_units: List[int] = None
    dropout: float = 0.2

    def __post_init__(self):
        if self.hidden_units is None:
            self.hidden_units = [256, 128]


@dataclass
class DataConfig:
    """Data configuration"""

    data_path: str = "./input/toss-next-challenge"
    target_col: str = "clicked"
    seq_col: str = "seq"
    exclude_cols: List[str] = None
    downsample_ratio: float = 2.0
    test_size: float = 0.2

    def __post_init__(self):
        if self.exclude_cols is None:
            self.exclude_cols = ["clicked", "seq", "ID"]


@dataclass
class TrainingConfig:
    """Training configuration"""

    model: ModelConfig = None
    data: DataConfig = None
    device: str = "cuda"

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()


def load_yaml(path: str):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return EasyDict(config)


def validate_experiment_config(args: argparse.ArgumentParser, config: EasyDict):
    if args.use_seq_feature and args.model not in [
        "dcn",
        "dcn_v2",
        "deepfm",
        "xdeepfm",
        "ffm",
        "fm",
    ]:
        raise ValueError(
            f"Model {args.model} does not support sequence feature. Please set --use_seq_feature to False."
        )


def replace_config_with_args(
    args: argparse.ArgumentParser, config: EasyDict, result_path: str
) -> EasyDict:
    config.common.batch_size = args.batch_size
    config.data.result_path = result_path
    config.data.fold_idx_for_fm = args.fold_idx_for_fm
    config.data.use_seq_feature = args.use_seq_feature
    return config
