import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List
from models.fm.ffm.ffm_base import FFMBase
from layers import MultiHeadAttentionWithAggregation


class Model(FFMBase):
    """
    Field-aware Factorization Machine inheriting from LogisticRegression

    FFM formula: ŷ = w₀ + Σwᵢxᵢ + ΣᵢΣⱼ>ⱼ⟨vᵢ,fⱼ,vⱼ,fᵢ⟩xᵢxⱼ

    Each feature has separate embeddings for each field it interacts with.
    """

    def __init__(
        self,
        categorical_field_dims: Optional[List[int]] = None,
        numerical_field_count: int = 0,
        embed_dim: int = 10,
        vocab_size: int = 0,
        dropout_rate: float = 0.2,
        d_model: int = 128,
        nhead: int = 8,
        use_causal_mask: bool = False,
        max_seq_length: int = 512,
        **kwargs,
    ):
        # Initialize parent class (gets bias + first-order interactions)
        super().__init__(
            categorical_field_dims=categorical_field_dims,
            numerical_field_count=numerical_field_count,
            embed_dim=embed_dim,
        )

        self.encoder = MultiHeadAttentionWithAggregation(
            vocab_size=vocab_size,
            fm_embedding_dim=embed_dim,
            d_model=d_model,
            nhead=nhead,
            max_seq_len=max_seq_length,
            use_feedforward=False,
            use_causal_mask=use_causal_mask,
            aggregation="attention_pool",
        )

        self.output_layer = nn.Sequential(
            *[
                nn.Linear(embed_dim + 1, 16),  # plus one for first seq embedding
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(16, 1),
            ]
        )

        # Add mlp layer for efficient tensor processing
        self.reduce_mlp = nn.Sequential(
            *[
                nn.Linear(self.total_field_count * embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]
        )

    def forward(
        self,
        numerical_x: Optional[Tensor] = None,
        categorical_x: Optional[Tensor] = None,
        seq: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Forward pass of FFM: LR + field-aware second-order interactions

        Args:
            categorical_x: Categorical features (batch_size, num_categorical)
            numerical_x: Numerical features (batch_size, num_numerical)

        Returns:
            FFM predictions (batch_size, 1)
        """
        numerical_x = self.bn_num(numerical_x)

        batch_size = (
            categorical_x.size(0) if categorical_x is not None else numerical_x.size(0)
        )

        # Start with bias term (w₀)
        output = self.bias.expand(batch_size).clone()

        # sequence encoding
        seq_emb = self.encoder(seq)  # (batch_size, embed_dim)

        # Add first-order interactions (Σwᵢxᵢ)
        output += self._first_order_interactions(numerical_x, categorical_x)

        batch_size = (
            categorical_x.size(0) if categorical_x is not None else numerical_x.size(0)
        )
        device = next(self.parameters()).device

        # Get all feature values
        x_all = self._get_all_x(batch_size, categorical_x, numerical_x, device)

        # Get all embeddings
        all_embeddings = self._get_all_embeddings(
            batch_size, categorical_x, numerical_x, device
        )

        # Add field-aware second-order interactions
        output += self._field_aware_interactions(x_all, all_embeddings, device)

        # Combine cross and deep outputs
        combined_output = torch.cat([output.unsqueeze(1), seq_emb], dim=1)

        # Final prediction
        output = self.output_layer(combined_output)

        return output

    def _field_aware_interactions(
        self, x_all: Tensor, all_embeddings: Tensor, device: str
    ) -> Tensor:
        """
        Compute field-aware second-order interactions: ΣᵢΣⱼ>ⱼ⟨vᵢ,fⱼ,vⱼ,fᵢ⟩xᵢxⱼ
        Using torch.einsum for efficient computation.
        """
        # Create upper triangular mask for i < j interactions
        field_indices = torch.arange(self.total_field_count, device=device)
        i_mask, j_mask = torch.meshgrid(field_indices, field_indices, indexing="ij")
        upper_tri_mask = i_mask < j_mask

        # Use einsum to compute all dot products efficiently
        # 'bijd,bjid->bij' means: for each batch b, compute dot product between
        # embeddings at positions (i,j,d) and (j,i,d) across dimension d
        dot_products = torch.einsum("bijd,bjid->bij", all_embeddings, all_embeddings)

        # Compute all pairwise x_i * x_j products
        x_products = torch.einsum("bi,bj->bij", x_all, x_all)

        # Combine dot products with x products and apply upper triangular mask
        interactions = dot_products * x_products * upper_tri_mask.float()

        # Sum all valid interactions
        interaction_sum = torch.sum(interactions, dim=(1, 2))

        # Clamp to prevent extreme values
        interaction_sum = torch.clamp(interaction_sum, -100, 100)

        return interaction_sum
