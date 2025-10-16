from typing import List

import numpy as np
import polars as pl


def write_seq_to_txt_file(train: pl.DataFrame, file_path: str):
    """
    Write sequences from DataFrame to text file efficiently.
    Each sequence (comma-separated string) is converted to space-separated and written as a separate line.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        # Split comma-separated strings and join with spaces
        for seq_str in train.select(
            pl.col("seq").str.replace_all(" ", "").str.replace_all(",", " ")
        ).iter_rows():
            f.write(seq_str[0].strip() + "\n")

    return


class SequenceTruncator:
    """Smart truncation strategies for long sequences while preserving ordering meaning"""

    def __init__(self, max_length: int = 128):
        self.max_length = max_length

    def simple_truncation(self, sequence: List[int], method: str = "last") -> List[int]:
        """
        Simple truncation strategies

        Args:
            sequence: Input sequence
            method: 'first', 'last', 'middle'
        """
        if len(sequence) <= self.max_length:
            return sequence

        if method == "last":
            return sequence[-self.max_length :]
        elif method == "first":
            return sequence[: self.max_length]
        elif method == "middle":
            start = (len(sequence) - self.max_length) // 2
            return sequence[start : start + self.max_length]
        else:
            raise ValueError(f"Unknown method: {method}")

    def hierarchical_sampling(
        self, sequence: List[int], recent_ratio: float = 0.7
    ) -> List[int]:
        """
        Sample more from recent history, less from distant past
        GUARANTEED to return exactly max_length items (or original length if shorter)

        Args:
            sequence: Input sequence
            recent_ratio: Proportion of max_length to allocate to recent items
        """
        if len(sequence) <= self.max_length:
            return sequence

        recent_count = int(self.max_length * recent_ratio)
        historical_count = self.max_length - recent_count

        # Always take the most recent items
        recent_items = sequence[-recent_count:]

        # Sample from historical items (excluding recent ones)
        historical_items = sequence[:-recent_count]

        if len(historical_items) <= historical_count:
            # If we have fewer historical items than allocated slots,
            # redistribute the extra slots to recent items
            sampled_historical = historical_items
            extra_slots = historical_count - len(historical_items)

            # Take more recent items to fill up to max_length
            total_recent_needed = recent_count + extra_slots
            recent_items = sequence[-total_recent_needed:]
        else:
            # Uniform sampling from historical items (stepwise sampling not based on probability)
            indices = np.linspace(
                0, len(historical_items) - 1, historical_count, dtype=int
            )
            sampled_historical = historical_items[indices]

        result = np.concatenate([sampled_historical, recent_items])

        # Double-check: ensure we return exactly max_length items
        assert len(result) == self.max_length, (
            f"Expected {self.max_length}, got {len(result)}"
        )

        return result

    def exponential_sampling(
        self, sequence: List[int], decay_factor: float = 0.95
    ) -> List[int]:
        """
        Sample with exponentially higher probability for recent items.
        Note that this could take longer time because of random sampling.

        Args:
            sequence: Input sequence
            decay_factor: Exponential decay factor (higher = more recent bias)
        """
        if len(sequence) <= self.max_length:
            return sequence

        # Create exponential weights (recent items have higher weights)
        positions = np.arange(len(sequence))
        weights = decay_factor ** (len(sequence) - 1 - positions)
        weights = weights / weights.sum()  # Normalize

        # Sample without replacement
        selected_indices = np.random.choice(
            len(sequence), size=self.max_length, replace=False, p=weights
        )
        selected_indices = np.sort(selected_indices)  # Maintain order

        return [sequence[i] for i in selected_indices]
