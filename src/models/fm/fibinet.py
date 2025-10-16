from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from models.fm.lr import Model as LogisticRegression
from layers import SENetBlock, BilinearInteraction


class Model(LogisticRegression):
    """
    FiBiNet: Feature Importance and Bilinear Feature Interaction Network

    FiBiNet combines:
    1. First-order interactions (from LogisticRegression)
    2. SENET-enhanced embeddings for feature importance
    3. Bilinear interactions between field pairs
    """

    def __init__(
        self,
        categorical_field_dims: List[int] = None,
        numerical_field_count: int = 0,
        embed_dim: int = 10,
        reduction_ratio: float = 3,
        **kwargs,
    ):
        # Initialize parent class (gets bias + first-order interactions)
        super(Model, self).__init__(categorical_field_dims, numerical_field_count)

        self.embed_dim = embed_dim
        self.reduction_ratio = reduction_ratio

        # Calculate total number of fields
        self.total_fields = self.num_categorical + self.numerical_field_count

        # Add embeddings
        if self.num_categorical > 0:
            self._setup_categorical_embeddings()

        if self.numerical_field_count > 0:
            self._setup_numerical_embeddings()

        # SENET block for feature importance
        if self.total_fields > 0:
            self.senet = SENetBlock(self.total_fields, reduction_ratio)

            # Bilinear interaction layer
            self.bilinear = BilinearInteraction(embed_dim, self.total_fields)

            # Final MLP for bilinear interactions
            self.bilinear_mlp = nn.Sequential(
                nn.Linear(self.bilinear.num_interactions, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 1),
            )

        self._init_embedding_weights()

    def _setup_categorical_embeddings(self):
        """Setup categorical embeddings"""
        total_vocab_size = self.field_offsets[-1]
        self.categorical_embeddings = nn.Embedding(total_vocab_size, self.embed_dim)

    def _setup_numerical_embeddings(self):
        """Setup numerical embeddings"""
        self.numerical_embeddings = nn.Parameter(
            torch.randn(self.numerical_field_count, self.embed_dim)
        )

    def _init_embedding_weights(self):
        """Initialize embedding weights"""
        if hasattr(self, "categorical_embeddings"):
            nn.init.xavier_normal_(self.categorical_embeddings.weight, gain=1.0)

        if hasattr(self, "numerical_embeddings"):
            nn.init.xavier_normal_(self.numerical_embeddings, gain=1.0)

        if hasattr(self, "bilinear"):
            nn.init.xavier_normal_(self.bilinear.bilinear_weights, gain=1.0)

    def forward(
        self, numerical_x: Tensor = None, categorical_x: Tensor = None, **kwargs
    ):
        """
        Forward pass of FiBiNet

        Args:
            categorical_x: Categorical features (batch_size, num_categorical)
            numerical_x: Numerical features (batch_size, num_numerical)

        Returns:
            FiBiNet predictions (batch_size, 1)
        """
        if numerical_x is not None:
            numerical_x = self.bn_num(numerical_x)

        batch_size = (
            categorical_x.size(0) if categorical_x is not None else numerical_x.size(0)
        )

        # Start with bias term
        output = self.bias.expand(batch_size).clone()

        # Add first-order interactions
        output += self._first_order_interactions(numerical_x, categorical_x)

        # Add FiBiNet interactions if we have fields
        if self.total_fields > 0:
            output += self._fibinet_interactions(numerical_x, categorical_x)

        return output.unsqueeze(-1)  # (batch_size, 1)

    def _fibinet_interactions(self, numerical_x: Tensor, categorical_x: Tensor):
        """Compute FiBiNet interactions: SENET + Bilinear"""
        batch_size = (
            categorical_x.size(0) if categorical_x is not None else numerical_x.size(0)
        )
        device = next(self.parameters()).device

        # Collect all embeddings
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
            # Weight by numerical values
            num_embeddings = num_embeddings * numerical_x.unsqueeze(-1)
            all_embeddings.append(num_embeddings)

        if not all_embeddings:
            return torch.zeros(batch_size, device=device)

        # Concatenate all embeddings
        embeddings = torch.cat(
            all_embeddings, dim=1
        )  # (batch_size, total_fields, embed_dim)

        # Apply SENET for feature importance
        enhanced_embeddings = self.senet(embeddings)

        # Compute bilinear interactions
        bilinear_output = self.bilinear(
            enhanced_embeddings
        )  # (batch_size, num_interactions)

        # Final MLP
        fibinet_output = self.bilinear_mlp(bilinear_output).squeeze(-1)  # (batch_size,)

        return fibinet_output
