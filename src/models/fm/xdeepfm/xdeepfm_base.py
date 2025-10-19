from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from models.fm.deepfm.deepfm_base import DeepFMBase
from layers import CIN


class xDeepFMBase(DeepFMBase):
    """
    xDeepFM (Extreme Deep Factorization Machine) inheriting from DeepFM

    xDeepFM formula: ŷ = Linear + CIN + DNN
                      = (w₀ + Σwᵢxᵢ) + CIN_output + DNN_output

    Components:
    1. Linear: Inherits from LogisticRegression (w₀ + Σwᵢxᵢ)
    2. CIN: Explicit vector-wise interactions (replaces FM's 2nd-order)
    3. DNN: Implicit high-order interactions (inherited from DeepFM)
    """

    def __init__(
        self,
        categorical_field_dims: List[int] = None,
        numerical_field_count: int = 0,
        embed_dim: int = 16,
        cin_layer_dims: List[int] = [128, 64],
        mlp_dims: List[int] = [512, 256, 128],
        dropout: float = 0.2,
        use_seq_feature=False,
        **kwargs,
    ):
        # Initialize DeepFM (gets Linear + DNN components)
        super().__init__(
            categorical_field_dims=categorical_field_dims,
            numerical_field_count=numerical_field_count,
            embed_dim=embed_dim,
            mlp_dims=mlp_dims,
            dropout=dropout,
            use_seq_feature=use_seq_feature,
        )

        self.cin_layer_dims = cin_layer_dims
        if self.cin_layer_dims[-1] != embed_dim:
            self.cin_layer_dims.append(
                embed_dim
            )  # Ensure last CIN layer matches embed_dim

        # Total number of fields for CIN
        total_fields = (
            self.num_categorical
            + self.numerical_field_count
            + (1 if use_seq_feature else 0)
        )

        # CIN component for explicit high-order interactions
        if total_fields > 0 and cin_layer_dims:
            self.cin = CIN(total_fields, cin_layer_dims, embed_dim)

            # Output layer for CIN component
            self.cin_output = nn.Linear(self.cin.output_dim, 1, bias=False)
        else:
            self.cin = None
            self.cin_output = None

        self._init_cin_weights()

    def _init_cin_weights(self):
        """Initialize CIN component weights"""
        if self.cin_output is not None:
            nn.init.xavier_uniform_(self.cin_output.weight, gain=1.5)

    def _get_linear_component(self, numerical_x: Tensor, categorical_x: Tensor):
        """Get linear component from LogisticRegression"""
        batch_size = (
            categorical_x.size(0) if categorical_x is not None else numerical_x.size(0)
        )

        # Get bias + first-order interactions
        output = self.bias.expand(batch_size).clone()
        output += self._first_order_interactions(numerical_x, categorical_x)

        return output.unsqueeze(-1)  # (batch_size, 1)

    def _get_cin_component(
        self, numerical_x: Tensor, categorical_x: Tensor, seq_emb: Tensor = None
    ):
        """Get CIN component output"""
        if self.cin is None:
            batch_size = (
                categorical_x.size(0)
                if categorical_x is not None
                else numerical_x.size(0)
            )
            device = next(self.parameters()).device
            return torch.zeros(batch_size, 1, device=device)

        # Get all embeddings including seq embedding
        embeddings = self._get_all_embeddings(numerical_x, categorical_x, seq_emb)

        if embeddings is None:
            batch_size = (
                categorical_x.size(0)
                if categorical_x is not None
                else numerical_x.size(0)
            )
            device = next(self.parameters()).device
            return torch.zeros(batch_size, 1, device=device)

        # Pass through CIN
        cin_features = self.cin(embeddings)  # (batch_size, cin_output_dim)

        # Final linear transformation
        cin_output = self.cin_output(cin_features)  # (batch_size, 1)

        return cin_output
