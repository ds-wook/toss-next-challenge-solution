from __future__ import annotations

from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig

from features.sequence import build_seq_stats


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    train = pl.read_parquet(Path(cfg.data.path) / "train.parquet")
    train = build_seq_stats(train)

    train.write_parquet(Path(cfg.data.path) / "train_preprocessed.parquet")
    del train

    test = pl.read_parquet(Path(cfg.data.path) / "test.parquet")
    test = build_seq_stats(test)

    test.write_parquet(Path(cfg.data.path) / "test_preprocessed.parquet")


if __name__ == "__main__":
    _main()
