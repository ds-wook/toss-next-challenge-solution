from typing import List

import torch.nn as nn
import torch


class Model(nn.Module):
    """
    Logistic Regression model supporting both categorical and numerical features

    LR formula: ŷ = sigmoid(w₀ + Σwᵢxᵢ)

    Where:
    - w₀ is global bias (intercept)
    - wᵢ are feature weights

    Args:
        categorical_field_dims: List of vocabulary sizes for categorical features
        numerical_field_count: Number of numerical features
    """

    def __init__(
        self,
        categorical_field_dims: List[int] = None,
        numerical_field_count: int = 0,
        **kwargs,
    ):
        super(Model, self).__init__()

        self.categorical_field_dims = categorical_field_dims or []
        self.numerical_field_count = numerical_field_count
        self.num_categorical = len(self.categorical_field_dims)

        # batch norm for numerical features
        self.bn_num = nn.BatchNorm1d(self.numerical_field_count)

        # Global bias term (w₀ - intercept)
        self.bias = nn.Parameter(torch.zeros(1))

        # Categorical features - Vectorized approach
        if self.num_categorical > 0:
            self._setup_categorical_features()

        # Numerical features
        if numerical_field_count > 0:
            self._setup_numerical_features()

        self._init_weights()

    def _setup_categorical_features(self):
        """Setup categorical feature embeddings and offsets"""
        # Create offset mapping for vectorized embedding lookup
        # Each value in offset indicates the starting index of that feature's embeddings,
        # which means number of unique categories for just previous feature
        self.field_offsets = [0]
        total_vocab_size = 0

        for dim in self.categorical_field_dims:
            total_vocab_size += dim
            self.field_offsets.append(total_vocab_size)

        # Single large embedding table for categorical weights
        self.categorical_linear = nn.Embedding(total_vocab_size, 1)

        # Register field_offsets as buffer
        self.register_buffer(
            "field_offsets_tensor",
            torch.tensor(self.field_offsets[:-1], dtype=torch.long),
        )

    def _setup_numerical_features(self):
        """Setup numerical feature weights"""
        # Linear weights (wᵢ) for numerical features
        self.numerical_linear = nn.Linear(self.numerical_field_count, 1, bias=False)

    def _init_weights(self):
        """Initialize model weights"""
        # Initialize bias to zero
        nn.init.zeros_(self.bias)

        # Initialize categorical weights
        if hasattr(self, "categorical_linear"):
            nn.init.xavier_uniform_(self.categorical_linear.weight, gain=1.0)

        # Initialize numerical weights
        if hasattr(self, "numerical_linear"):
            nn.init.xavier_uniform_(self.numerical_linear.weight, gain=1.0)

    def forward(self, numerical_x=None, categorical_x=None, **kwargs):
        """
        Forward pass of Logistic Regression

        Args:
            categorical_x: Categorical features (batch_size, num_categorical)
            numerical_x: Numerical features (batch_size, num_numerical)

        Returns:
            Logits (batch_size, 1)
        """
        if categorical_x is None and numerical_x is None:
            raise ValueError(
                "At least one of categorical_x or numerical_x must be provided"
            )

        batch_size = (
            categorical_x.size(0) if categorical_x is not None else numerical_x.size(0)
        )

        # Start with bias term (w₀)
        output = self.bias.expand(batch_size).clone()

        # Add first-order interactions (Σwᵢxᵢ)
        output += self._first_order_interactions(
            self.bn_num(numerical_x), categorical_x
        )

        return output.unsqueeze(-1)  # (batch_size, 1)

    def _first_order_interactions(self, numerical_x, categorical_x):
        """
        Compute first-order interactions: Σwᵢxᵢ

        Args:
            categorical_x: Categorical features
            numerical_x: Numerical features

        Returns:
            First-order interaction terms (batch_size,)
        """
        batch_size = (
            categorical_x.size(0) if categorical_x is not None else numerical_x.size(0)
        )
        device = next(self.parameters()).device

        first_order = torch.zeros(batch_size, device=device)

        # Categorical first-order: Σwᵢ (since xᵢ=1 for one-hot encoded categorical)
        if categorical_x is not None and self.num_categorical > 0:
            # Add offsets to convert local indices to global indices
            global_indices = categorical_x + self.field_offsets_tensor.unsqueeze(0)

            # Single embedding lookup for all categorical features
            cat_linear_weights = self.categorical_linear(
                global_indices
            )  # (batch_size, num_categorical, 1)
            first_order += cat_linear_weights.sum(dim=(1, 2))

        # Numerical first-order: Σwᵢxᵢ
        if numerical_x is not None and self.numerical_field_count > 0:
            num_linear_weights = self.numerical_linear(numerical_x)
            first_order += num_linear_weights.squeeze(-1)

        return first_order
