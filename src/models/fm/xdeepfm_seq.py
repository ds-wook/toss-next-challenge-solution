from typing import List

import torch
import torch.nn as nn

from models.fm.deepfm_seq import Model as DeepFMWithSequence
from layers import CIN


class Model(DeepFMWithSequence):
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
        mlp_dims: List[int] = [128, 64],
        dropout: float = 0.2,
        vocab_size: int = 0,
        d_model: int = 128,
        nhead: int = 8,
        use_causal_mask: bool = False,
        max_seq_length: int = 512,
        **kwargs,
    ):
        # Initialize DeepFM (gets Linear + DNN components)
        super(Model, self).__init__(
            categorical_field_dims=categorical_field_dims,
            numerical_field_count=numerical_field_count,
            embed_dim=embed_dim,
            mlp_dims=mlp_dims,
            dropout=dropout,
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            max_seq_length=max_seq_length,
            use_causal_mask=use_causal_mask,
        )

        self.cin_layer_dims = cin_layer_dims
        if self.cin_layer_dims[-1] != embed_dim:
            self.cin_layer_dims.append(
                embed_dim
            )  # Ensure last CIN layer matches embed_dim

        # Total number of fields for CIN
        # one is added because of sequence feature
        total_fields = self.num_categorical + self.numerical_field_count + 1

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

    def forward(self, numerical_x=None, categorical_x=None, seq=None, **kwargs):
        """
        Forward pass of xDeepFM: Linear + CIN + DNN

        Args:
            categorical_x: Categorical features (batch_size, num_categorical)
            numerical_x: Numerical features (batch_size, num_numerical)

        Returns:
            xDeepFM predictions (batch_size, 1)
        """
        numerical_x = self.bn_num(numerical_x)

        # 1. Linear component (w₀ + Σwᵢxᵢ) - inherited from LogisticRegression
        linear_output = self._get_linear_component(numerical_x, categorical_x)

        # sequence encoding
        seq_emb = self.encoder(seq)  # (batch_size, embed_dim)

        # 2. CIN component (explicit vector-wise interactions)
        cin_output = self._get_cin_component(numerical_x, categorical_x, seq_emb)

        # 3. DNN component (implicit high-order interactions) - inherited from DeepFM
        dnn_output = self._deep_component(numerical_x, categorical_x, seq_emb)

        # Combine all three components
        return linear_output + cin_output + dnn_output

    def _get_linear_component(self, numerical_x, categorical_x):
        """Get linear component from LogisticRegression"""
        batch_size = (
            categorical_x.size(0) if categorical_x is not None else numerical_x.size(0)
        )

        # Get bias + first-order interactions
        output = self.bias.expand(batch_size).clone()
        output += self._first_order_interactions(numerical_x, categorical_x)

        return output.unsqueeze(-1)  # (batch_size, 1)

    def _get_cin_component(self, numerical_x, categorical_x, seq_emb):
        """Get CIN component output"""
        if self.cin is None:
            batch_size = (
                categorical_x.size(0)
                if categorical_x is not None
                else numerical_x.size(0)
            )
            device = next(self.parameters()).device
            return torch.zeros(batch_size, 1, device=device)

        # Get all embeddings (same as used in DeepFM)
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

    def _get_all_embeddings(self, numerical_x, categorical_x, seq_emb):
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
            all_embeddings.append(num_embeddings)

        # sequence embeddings
        all_embeddings.append(seq_emb.unsqueeze(1))

        if not all_embeddings:
            return None

        # Concatenate all embeddings
        embeddings = torch.cat(
            all_embeddings, dim=1
        )  # (batch_size, total_fields, embed_dim)

        return embeddings
