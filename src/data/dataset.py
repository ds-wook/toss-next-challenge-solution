"""
Dataset classes for CTR prediction
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torch import Tensor
import random
import os

from utils.seq_embedding import SequenceTruncator


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ClickDataset(Dataset):
    """Custom dataset for CTR prediction with sequence data"""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None,
        num_cols: List[str] = None,
        cat_cols: List[str] = None,
        seq_col: str = "seq",
        target_col: Optional[str] = None,
        has_target: bool = True,
        max_seq_length: int = 512,
        seq_split_strategy: str = "front_random",
        seq_split_ratio: float = 0.7,
    ):
        """
        Args:
            df: Input dataframe
            feature_cols: List of all feature column names (legacy mode)
            num_cols: List of numerical feature column names (new mode)
            cat_cols: List of categorical feature column names (new mode)
            seq_col: Sequence column name
            target_col: Target column name
            has_target: Whether dataset has target labels
            max_seq_length: Maximum sequence length
            seq_split_strategy: Strategy for splitting long sequences
                - "front": Take only front part
                - "back": Take only back part
                - "front_back": Take front and back parts
                - "front_random": Take front part + random from remaining (default)
                - "back_random": Take back part + random from remaining
            seq_split_ratio: Ratio for front/back split (default: 0.7)
        """
        self.df = df.reset_index(drop=True)
        self.seq_col = seq_col
        self.target_col = target_col
        self.has_target = has_target
        self.max_seq_length = max_seq_length
        self.seq_split_strategy = seq_split_strategy
        self.seq_split_ratio = seq_split_ratio

        # Handle both legacy and new modes
        if feature_cols is not None:
            # Legacy mode: all features in one tensor
            self.feature_cols = feature_cols
            self.num_cols = None
            self.cat_cols = None
            self.X = self.df[self.feature_cols].astype(float).fillna(0).values
        else:
            # New mode: separate numerical and categorical features
            self.feature_cols = None
            self.num_cols = num_cols or []
            self.cat_cols = cat_cols or []

            # Pre-compute features for efficiency
            if self.num_cols:
                self.num_X = self.df[self.num_cols].astype(float).fillna(0).values
            else:
                self.num_X = None

            if self.cat_cols:
                self.cat_X = self.df[self.cat_cols].astype(int).values
            else:
                self.cat_X = None

        # Store sequence strings for lazy parsing
        self.seq_strings = self.df[self.seq_col].astype(str).values

        if self.has_target:
            self.y = self.df[self.target_col].astype(np.float32).values

    def __len__(self):
        return len(self.df)

    def _apply_sequence_length_limit(self, arr: np.ndarray) -> np.ndarray:
        """Apply sequence length limiting based on strategy"""
        if len(arr) <= self.max_seq_length:
            return arr

        if self.seq_split_strategy == "front":
            return arr[: self.max_seq_length]

        elif self.seq_split_strategy == "back":
            return arr[-self.max_seq_length :]

        elif self.seq_split_strategy == "front_back":
            front_length = int(self.max_seq_length * self.seq_split_ratio)
            back_length = self.max_seq_length - front_length

            front_part = arr[:front_length]
            back_part = (
                arr[-back_length:]
                if back_length > 0
                else np.array([], dtype=np.float32)
            )

            return np.concatenate([front_part, back_part])

        elif self.seq_split_strategy == "front_random":
            front_length = int(self.max_seq_length * self.seq_split_ratio)
            remaining_length = self.max_seq_length - front_length

            # Front part
            front_part = arr[:front_length]

            # Random selection from remaining part
            if len(arr) > front_length and remaining_length > 0:
                back_start_idx = front_length
                back_end_idx = len(arr)

                if back_end_idx - back_start_idx >= remaining_length:
                    max_start = back_end_idx - remaining_length
                    random_start = np.random.randint(back_start_idx, max_start + 1)
                    back_part = arr[random_start : random_start + remaining_length]
                else:
                    back_part = arr[back_start_idx:]
            else:
                back_part = np.array([], dtype=np.float32)

            return np.concatenate([front_part, back_part])

        elif self.seq_split_strategy == "back_random":
            back_length = int(self.max_seq_length * self.seq_split_ratio)
            remaining_length = self.max_seq_length - back_length

            # Back part
            back_part = (
                arr[-back_length:]
                if back_length > 0
                else np.array([], dtype=np.float32)
            )

            # Random selection from remaining part
            if len(arr) > back_length and remaining_length > 0:
                front_end_idx = len(arr) - back_length

                if front_end_idx >= remaining_length:
                    random_start = np.random.randint(
                        0, front_end_idx - remaining_length + 1
                    )
                    front_part = arr[random_start : random_start + remaining_length]
                else:
                    front_part = arr[:front_end_idx]
            else:
                front_part = np.array([], dtype=np.float32)

            return np.concatenate([front_part, back_part])

        else:
            # Default: just truncate
            return arr[: self.max_seq_length]

    def __getitem__(self, idx):
        # Handle features based on mode
        if self.feature_cols is not None:
            # Legacy mode: single feature tensor
            x = torch.tensor(self.X[idx], dtype=torch.float32)
            num_x = None
            cat_x = None
        else:
            # New mode: separate numerical and categorical features
            x = None
            if self.num_X is not None:
                num_x = torch.tensor(self.num_X[idx], dtype=torch.float)
            else:
                num_x = None

            if self.cat_X is not None:
                cat_x = torch.tensor(self.cat_X[idx], dtype=torch.long)
            else:
                cat_x = None

        # Parse sequence string
        s = self.seq_strings[idx]
        if s and s != "nan" and s != "":
            try:
                arr = np.fromstring(s, sep=",", dtype=np.float32)
            except Exception:
                arr = np.array([0.0], dtype=np.float32)
        else:
            arr = np.array([0.0], dtype=np.float32)

        if arr.size == 0:
            arr = np.array([0.0], dtype=np.float32)  # Handle empty sequences

        # Apply sequence length limiting
        arr = self._apply_sequence_length_limit(arr)

        # Additional safety: limit very long sequences more strictly
        if len(arr) > 5000:
            arr = arr[:5000]

        seq = torch.from_numpy(arr)

        if self.has_target:
            y = torch.tensor(self.y[idx], dtype=torch.float32)

            # Return format based on mode
            if self.feature_cols is not None:
                return x, seq, y
            else:
                # Ensure num_x and cat_x are always tensors, even if empty
                if num_x is None:
                    num_x = torch.tensor([], dtype=torch.float32)
                if cat_x is None:
                    cat_x = torch.tensor([], dtype=torch.long)
                return num_x, cat_x, seq, y
        else:
            # Return format based on mode
            if self.feature_cols is not None:
                return x, seq
            else:
                # Ensure num_x and cat_x are always tensors, even if empty
                if num_x is None:
                    num_x = torch.tensor([], dtype=torch.float32)
                if cat_x is None:
                    cat_x = torch.tensor([], dtype=torch.long)
                return num_x, cat_x, seq


class StratifiedBatchSampler(Sampler):
    """
    Sampler that ensures each batch has a specified ratio of positive/negative samples
    """

    def __init__(self, labels, batch_size, positive_ratio=0.1, shuffle=True, seed=None):
        """
        Args:
            labels: Target labels (0 or 1)
            batch_size: Size of each batch
            positive_ratio: Desired ratio of positive samples in each batch
            shuffle: Whether to shuffle samples
            seed: Random seed for reproducibility
        """
        self.labels = labels
        self.batch_size = batch_size
        self.positive_ratio = positive_ratio
        self.shuffle = shuffle
        self.seed = seed

        # Set seed if provided
        if self.seed is not None:
            np.random.seed(self.seed)

        # Get indices for positive and negative samples
        if isinstance(labels, torch.Tensor):
            labels_np = labels.numpy()
        else:
            labels_np = labels

        self.positive_indices = np.where(labels_np == 1)[0]
        self.negative_indices = np.where(labels_np == 0)[0]

        # Calculate samples per batch
        self.pos_per_batch = max(1, int(batch_size * positive_ratio))
        self.neg_per_batch = batch_size - self.pos_per_batch

        # Calculate number of batches
        max_pos_batches = len(self.positive_indices) // self.pos_per_batch
        max_neg_batches = len(self.negative_indices) // self.neg_per_batch
        self.num_batches = min(max_pos_batches, max_neg_batches)

        if self.num_batches == 0:
            raise ValueError("Not enough samples to create stratified batches")

    def __iter__(self):
        # Reset seed for each epoch to ensure consistent shuffling
        if self.seed is not None:
            np.random.seed(self.seed + getattr(self, "_epoch", 0))

        if self.shuffle:
            pos_indices = np.random.permutation(self.positive_indices)
            neg_indices = np.random.permutation(self.negative_indices)
        else:
            pos_indices = self.positive_indices.copy()
            neg_indices = self.negative_indices.copy()

        for i in range(self.num_batches):
            # Get batch indices
            pos_batch = pos_indices[
                i * self.pos_per_batch : (i + 1) * self.pos_per_batch
            ]
            neg_batch = neg_indices[
                i * self.neg_per_batch : (i + 1) * self.neg_per_batch
            ]

            # Combine and shuffle batch indices
            batch_indices = np.concatenate([pos_batch, neg_batch])
            if self.shuffle:
                np.random.shuffle(batch_indices)

            yield batch_indices.tolist()

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch):
        """Set epoch for different shuffling in each epoch"""
        self._epoch = epoch


def create_torch_data_loader(
    test_x: Tensor,
    train_x: Tensor = None,
    train_y: Tensor = None,
    val_x: Tensor = None,
    val_y: Tensor = None,
    train_seq: List[str] = None,
    val_seq: List[str] = None,
    test_seq: List[str] = None,
    batch_size: int = 128,
    test_only: bool = False,
    num_numeric_feat: int = None,
    num_cat_feat: int = None,
    num_workers: int = 4,
    stratified_sampling: bool = False,
    positive_ratio: float = 0.1,
    seed: int = None,
    use_seq_feature: bool = False,
    max_seq_length: int = 512,
    recent_ratio: float = 0.7,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Set seeds if provided
    if seed is not None:
        set_random_seeds(seed)

    if use_seq_feature:
        test_dataset = FMSeqDataset(
            X=test_x,
            num_numeric_feat=num_numeric_feat,
            num_cat_feat=num_cat_feat,
            seq=test_seq,
            is_inference=True,
            max_seq_length=max_seq_length,
            recent_ratio=recent_ratio,
        )
    else:
        test_dataset = FMDataset(
            X=test_x,
            num_numeric_feat=num_numeric_feat,
            num_cat_feat=num_cat_feat,
            is_inference=True,
        )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Note!!! We should NOT shuffle test data
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )

    if test_only:
        return test_dataloader

    if use_seq_feature:
        train_dataset = FMSeqDataset(
            X=train_x,
            y=train_y,
            num_numeric_feat=num_numeric_feat,
            num_cat_feat=num_cat_feat,
            seq=train_seq,
            is_inference=False,
            max_seq_length=max_seq_length,
            recent_ratio=recent_ratio,
        )
        val_dataset = FMSeqDataset(
            X=val_x,
            y=val_y,
            num_numeric_feat=num_numeric_feat,
            num_cat_feat=num_cat_feat,
            seq=val_seq,
            is_inference=False,
            max_seq_length=max_seq_length,
            recent_ratio=recent_ratio,
        )
    else:
        train_dataset = FMDataset(
            X=train_x,
            y=train_y,
            num_numeric_feat=num_numeric_feat,
            num_cat_feat=num_cat_feat,
            is_inference=False,
        )
        val_dataset = FMDataset(
            X=val_x,
            y=val_y,
            num_numeric_feat=num_numeric_feat,
            num_cat_feat=num_cat_feat,
            is_inference=False,
        )

    # Create stratified batch sampler for training if requested
    if stratified_sampling and train_y is not None:
        train_sampler = StratifiedBatchSampler(
            labels=train_y,
            batch_size=batch_size,
            positive_ratio=positive_ratio,
            shuffle=True,
            seed=seed,
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )
    else:
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )

    # For validation, you might also want stratified sampling
    if stratified_sampling and val_y is not None:
        val_sampler = StratifiedBatchSampler(
            labels=val_y,
            batch_size=batch_size,
            positive_ratio=positive_ratio,
            shuffle=False,  # Usually don't shuffle validation
            seed=seed,
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )
    else:
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )

    return train_dataloader, val_dataloader, test_dataloader


class FMDataset(Dataset):
    def __init__(
        self,
        X: Tensor,
        num_numeric_feat: int,
        num_cat_feat: int,
        y: Tensor = None,
        is_inference: bool = False,
    ):
        self.X = X
        self.num_numeric_feat = num_numeric_feat
        self.num_cat_feat = num_cat_feat
        self.y = y
        self.is_inference = is_inference

        assert self.X.size(1) == self.num_numeric_feat + self.num_cat_feat

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        num_x, cat_x = (
            self.X[idx, : self.num_numeric_feat],
            self.X[idx, self.num_numeric_feat :].long(),
        )
        if not self.is_inference:
            return num_x, cat_x, self.y[idx]
        else:
            return num_x, cat_x, torch.tensor(-1, dtype=torch.float32)


class FMSeqDataset(Dataset):
    def __init__(
        self,
        X: Tensor,
        num_numeric_feat: int,
        num_cat_feat: int,
        seq: List[str],
        max_seq_length: int = 512,
        recent_ratio: float = 0.7,
        y: Tensor = None,
        is_inference: bool = False,
    ):
        self.X = X
        self.num_numeric_feat = num_numeric_feat
        self.num_cat_feat = num_cat_feat
        self.seq = seq
        self.max_seq_length = max_seq_length
        self.recent_ratio = recent_ratio
        self.y = y
        self.is_inference = is_inference
        self.truncator = SequenceTruncator(max_length=max_seq_length)

        # note: seq feature is separately handled, so no need to check here
        assert self.X.size(1) == self.num_numeric_feat + self.num_cat_feat

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        num_x, cat_x = (
            self.X[idx, : self.num_numeric_feat],
            self.X[idx, self.num_numeric_feat :].long(),
        )

        # sequence parsing
        s = self.seq[idx]
        seq_arr = np.fromstring(s, sep=",", dtype=np.float32)
        # set length to max_seq_length using hierarchical sampling
        seq_arr_truncated = self.truncator.hierarchical_sampling(
            seq_arr, recent_ratio=self.recent_ratio
        )
        seq = torch.from_numpy(seq_arr_truncated).long()

        # Pad or truncate to max_seq_length using torch
        if seq.size(0) < self.max_seq_length:
            seq = F.pad(seq, (0, self.max_seq_length - seq.size(0)), "constant", 0)

        if not self.is_inference:
            return num_x, cat_x, seq, self.y[idx]
        else:
            return num_x, cat_x, seq, torch.tensor(-1, dtype=torch.float32)


class FMMixedDataset(Dataset):
    def __init__(
        self,
        num_x: Tensor,
        cat_sparse_x: Tensor,
        num_original_cat_feats: int,
        y: Tensor = None,
        is_inference: bool = False,
    ):
        self.num_x = num_x
        self.cat_sparse_x = cat_sparse_x
        self.num_original_cat_feats = num_original_cat_feats
        self.y = y
        self.is_inference = is_inference

    def __len__(self):
        return self.num_x.size(0)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # Extract sparse row using CSR format (more efficient for row access)
        sparse_row = self.cat_sparse_x[idx]  # CSR allows efficient row slicing

        # Return indices and values directly - no tensor creation needed
        col_indices = torch.from_numpy(sparse_row.indices).long()
        row_values = torch.from_numpy(sparse_row.data).float()

        if self.num_original_cat_feats < len(col_indices):
            raise ValueError(
                f"num_original_cat_feats ({self.num_original_cat_feats}) is less than the number of non-zero categorical features ({len(col_indices)}) in the sample."
            )

        if self.num_original_cat_feats > len(col_indices):
            # Pad with zeros if fewer non-zero features than original features
            col_indices = F.pad(
                col_indices,
                (0, self.num_original_cat_feats - len(col_indices)),
                "constant",
                0,
            )
            row_values = F.pad(
                row_values,
                (0, self.num_original_cat_feats - len(row_values)),
                "constant",
                0,
            )

        if self.is_inference:
            return (
                self.num_x[idx],
                (col_indices, row_values),
                torch.tensor(-1, dtype=torch.float32),
            )
        else:
            return self.num_x[idx], (col_indices, row_values), self.y[idx]


def sparse_collate_fn(batch: List):
    """Custom collate function to handle variable-length sparse data"""
    num_features, cat_features, labels = zip(*batch)

    # Stack numeric features and labels normally
    labels = torch.stack(labels)
    num_features = torch.stack(num_features)

    # For features, we need to handle variable lengths
    # So, pad to max length in batch
    max_nnz = max(len(feat[0]) for feat in cat_features)

    # Vectorized approach - create tensors directly
    batch_size = len(cat_features)
    batch_indices = torch.zeros((batch_size, max_nnz), dtype=torch.long)
    batch_values = torch.zeros((batch_size, max_nnz), dtype=torch.float32)

    # Fill tensors using advanced indexing
    for i, (indices, values) in enumerate(cat_features):
        seq_len = len(indices)
        if seq_len > 0:
            batch_indices[i, :seq_len] = indices
            batch_values[i, :seq_len] = values

    return num_features, (batch_indices, batch_values), labels


def collate_fn_train(batch):
    """Collate function for training data"""
    xs, seqs, ys = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    seqs_padded = torch.nn.utils.rnn.pad_sequence(
        seqs, batch_first=True, padding_value=0.0
    )
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)  # Prevent empty sequences
    return xs, seqs_padded, seq_lengths, ys


def collate_fn_infer(batch):
    """Collate function for inference data"""
    xs, seqs = zip(*batch)
    xs = torch.stack(xs)
    seqs_padded = torch.nn.utils.rnn.pad_sequence(
        seqs, batch_first=True, padding_value=0.0
    )
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return xs, seqs_padded, seq_lengths


def seq_collate_fn_train(batch):
    num_x, cat_x, seqs, ys = zip(*batch)

    # Handle empty tensors
    if len(num_x[0]) > 0:
        num_x = torch.stack(num_x)
    else:
        num_x = torch.empty(len(batch), 0, dtype=torch.float32)

    if len(cat_x[0]) > 0:
        cat_x = torch.stack(cat_x)
    else:
        cat_x = torch.empty(len(batch), 0, dtype=torch.long)

    ys = torch.stack(ys)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return num_x, cat_x, seqs_padded, seq_lengths, ys


def seq_collate_fn_infer(batch):
    num_x, cat_x, seqs = zip(*batch)

    # Handle empty tensors
    if len(num_x[0]) > 0:
        num_x = torch.stack(num_x)
    else:
        num_x = torch.empty(len(batch), 0, dtype=torch.float32)

    if len(cat_x[0]) > 0:
        cat_x = torch.stack(cat_x)
    else:
        cat_x = torch.empty(len(batch), 0, dtype=torch.long)

    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return num_x, cat_x, seqs_padded, seq_lengths
