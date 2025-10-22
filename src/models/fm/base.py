from abc import abstractmethod
from typing import List

import torch
from torch import nn, Tensor


class Base(nn.Module):
    def __init__(
        self,
        categorical_field_dims: List[int] = None,
        numerical_field_count: int = 0,
        **kwargs,
    ):
        super(Base, self).__init__()

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

    @abstractmethod
    def forward(**kwargs):
        raise NotImplementedError

    def _first_order_interactions(self, numerical_x: Tensor, categorical_x: Tensor):
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

    def _setup_categorical_embeddings(self):
        """Setup categorical embeddings for second-order interactions"""
        # Reuse the same total_vocab_size from parent class
        total_vocab_size = self.field_offsets[-1]
        self.categorical_embeddings = nn.Embedding(total_vocab_size, self.embed_dim)

    def _setup_numerical_embeddings(self):
        """Setup numerical embeddings for second-order interactions"""
        # Each numerical feature gets its own embed_dim-dimensional latent vector
        self.numerical_embeddings = nn.Parameter(
            torch.randn(self.numerical_field_count, self.embed_dim)
        )

    def _init_embedding_weights(self):
        """Initialize embedding weights for second-order interactions"""
        if hasattr(self, "categorical_embeddings"):
            nn.init.xavier_normal_(self.categorical_embeddings.weight, gain=1.0)

        if hasattr(self, "numerical_embeddings"):
            nn.init.xavier_normal_(self.numerical_embeddings, gain=1.0)

    def _get_all_embeddings(
        self,
        numerical_x: Tensor,
        categorical_x: Tensor,
        seq_emb: Tensor = None,
        is_num_weighted: bool = False,
    ):
        """Get all embeddings for CIN (same as DeepFM)"""
        batch_size = (
            categorical_x.size(0) if categorical_x is not None else numerical_x.size(0)
        )

        all_embeddings = []

        # Categorical embeddings
        if categorical_x is not None and self.num_categorical > 0:
            global_indices = categorical_x + self.field_offsets_tensor.unsqueeze(0)
            cat_embeddings = self.categorical_embeddings(global_indices)
            all_embeddings.append(cat_embeddings)

        # Numerical embeddings
        if numerical_x is not None and self.numerical_field_count > 0:
            num_embeddings = self.numerical_embeddings.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            if is_num_weighted:
                numerical_x_expanded = numerical_x.unsqueeze(-1)
                num_embeddings = num_embeddings * numerical_x_expanded
            all_embeddings.append(num_embeddings)

        # concat sequence embeddings if specified
        if seq_emb is not None:
            all_embeddings.append(seq_emb.unsqueeze(1))

        if not all_embeddings:
            return None

        # Concatenate all embeddings
        embeddings = torch.cat(
            all_embeddings, dim=1
        )  # (batch_size, total_fields, embed_dim)

        return embeddings

    def _get_all_x_values(
        self, numerical_x: Tensor, categorical_x: Tensor, use_seq_emb: bool = False
    ):
        batch_size = (
            categorical_x.size(0) if categorical_x is not None else numerical_x.size(0)
        )
        device = next(self.parameters()).device

        # Collect all embeddings and values in tensors (fully vectorized)
        all_x_values = []

        # Categorical embeddings and values
        if categorical_x is not None and self.num_categorical > 0:
            # For categorical features, x_i = 1
            cat_x_values = torch.ones(
                batch_size, self.num_categorical, 1, device=device, dtype=torch.float32
            )
            all_x_values.append(cat_x_values)

        # Numerical embeddings and values
        if numerical_x is not None and self.numerical_field_count > 0:
            num_x_values = numerical_x.unsqueeze(-1)
            all_x_values.append(num_x_values)

        # if using seq embedding, concat one vector
        if use_seq_emb:
            all_x_values.append(
                torch.ones(batch_size, 1, 1, device=device, dtype=torch.float32)
            )

        return torch.cat(all_x_values, dim=1)  # (batch_size, total_features, 1)
