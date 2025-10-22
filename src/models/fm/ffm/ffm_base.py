import torch
from torch import nn, Tensor
from typing import Optional, List

from models.fm.base import Base


class FFMBase(Base):
    def __init__(
        self,
        categorical_field_dims: Optional[List[int]] = None,
        numerical_field_count: int = 0,
        embed_dim: int = 10,
        **kwargs,
    ):
        # Initialize parent class (gets bias + first-order interactions)
        super().__init__(
            categorical_field_dims=categorical_field_dims,
            numerical_field_count=numerical_field_count,
        )

        self.embed_dim = embed_dim
        self.total_field_count = self.num_categorical + self.numerical_field_count

        # Add field-aware embeddings for second-order interactions
        if self.num_categorical > 0:
            self._setup_categorical_embeddings()

        if self.numerical_field_count > 0:
            self._setup_numerical_embeddings()

        self._init_embedding_weights()

    def _setup_categorical_embeddings(self):
        """Setup categorical field-aware embeddings"""
        # Each categorical feature has embeddings for each field it can interact with
        total_vocab_size = self.field_offsets[-1]
        self.categorical_embeddings = nn.ModuleList(
            [
                nn.Embedding(total_vocab_size, self.embed_dim)
                for _ in range(self.total_field_count)
            ]
        )

    def _setup_numerical_embeddings(self):
        """Setup numerical field-aware embeddings"""
        # Each numerical feature has embeddings for each field it can interact with
        self.numerical_embeddings = nn.Parameter(
            torch.randn(
                self.numerical_field_count, self.total_field_count, self.embed_dim
            )
        )

    def _init_embedding_weights(self):
        """Initialize field-aware embedding weights"""
        if hasattr(self, "categorical_embeddings"):
            for embedding in self.categorical_embeddings:
                nn.init.xavier_normal_(embedding.weight, gain=1.0)

        if hasattr(self, "numerical_embeddings"):
            nn.init.xavier_normal_(self.numerical_embeddings, gain=1.0)

        if hasattr(self, "output_layer"):
            for module in self.output_layer.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    nn.init.zeros_(module.bias)

        if hasattr(self, "reduce_mlp"):
            for module in self.reduce_mlp.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    nn.init.zeros_(module.bias)

    def _get_all_x(
        self,
        batch_size: int,
        categorical_x: Optional[Tensor],
        numerical_x: Optional[Tensor],
        device: str,
    ) -> Optional[Tensor]:
        """
        Prepare all feature values for field-aware interactions.

        Args:
            batch_size: Batch size
            categorical_x: Categorical features
            numerical_x: Numerical features
            device: Device to place tensors on

        Returns:
            x_all: Concatenated feature values (batch_size, total_fields) or None if no features
        """
        all_x_values = []

        # Categorical values (always 1 for one-hot)
        if categorical_x is not None and self.num_categorical > 0:
            cat_values = torch.ones(batch_size, self.num_categorical, device=device)
            all_x_values.append(cat_values)

        # Numerical values
        if numerical_x is not None and self.numerical_field_count > 0:
            all_x_values.append(numerical_x)

        if not all_x_values:
            return None

        return torch.cat(all_x_values, dim=1)  # (batch_size, total_fields)

    def _get_all_embeddings(
        self,
        batch_size: int,
        categorical_x: Optional[Tensor],
        numerical_x: Optional[Tensor],
        device,
    ) -> Tensor:
        """
        Pre-compute all embeddings for field-aware interactions.

        Args:
            batch_size: Batch size
            categorical_x: Categorical features
            numerical_x: Numerical features
            seq_emb: Sequence embeddings
            device: Device to place tensors on

        Returns:
            all_embeddings: Tensor of shape (batch_size, total_fields, total_fields, embed_dim)
        """
        # Pre-compute all embeddings more efficiently
        # Shape: (batch_size, total_fields, total_fields, embed_dim)
        all_embeddings = torch.zeros(
            batch_size,
            self.total_field_count,
            self.total_field_count,
            self.embed_dim,
            device=device,
        )

        # Fill categorical embeddings - vectorized
        if categorical_x is not None and self.num_categorical > 0:
            global_indices = categorical_x + self.field_offsets_tensor.unsqueeze(0)
            # Stack all categorical embeddings at once
            cat_embeddings_stack = torch.stack(
                [emb(global_indices) for emb in self.categorical_embeddings], dim=2
            )  # (batch_size, num_categorical, total_fields, embed_dim)
            all_embeddings[:, : self.num_categorical, :, :] = cat_embeddings_stack

        # Fill numerical embeddings - vectorized
        if numerical_x is not None and self.numerical_field_count > 0:
            num_start_idx = self.num_categorical
            # Expand numerical embeddings for batch
            num_emb_expanded = self.numerical_embeddings.unsqueeze(0).expand(
                batch_size, -1, -1, -1
            )  # (batch_size, num_numerical, total_fields, embed_dim)
            all_embeddings[:, num_start_idx:, :, :] = num_emb_expanded

        return all_embeddings
