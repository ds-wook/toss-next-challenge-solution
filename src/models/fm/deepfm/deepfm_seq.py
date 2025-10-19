import torch
from typing import Optional, List

from models.fm.deepfm.deepfm_base import DeepFMBase
from layers import MultiHeadAttentionWithAggregation


class Model(DeepFMBase):
    """
    DeepFM inheriting from DeepFMBase

    DeepFM formula: ŷ = FM_output + DNN_output
                     = (w₀ + Σwᵢxᵢ + Σᵢ<ⱼ⟨vᵢ,vⱼ⟩xᵢxⱼ) + DNN(embeddings)

    Inherits FM's bias + first-order + second-order interactions,
    adds deep neural network component for high-order interactions.
    """

    def __init__(
        self,
        categorical_field_dims: Optional[List[int]] = None,
        numerical_field_count: int = 0,
        embed_dim: int = 16,
        mlp_dims: List[int] = [512, 256, 128],
        dropout: float = 0.2,
        vocab_size: int = 0,
        d_model: int = 128,
        nhead: int = 8,
        use_causal_mask: bool = False,
        max_seq_length: int = 512,
        **kwargs,
    ):
        # Initialize parent FM class (gets all FM functionality)
        super(Model, self).__init__(
            categorical_field_dims=categorical_field_dims,
            numerical_field_count=numerical_field_count,
            embed_dim=embed_dim,
            mlp_dims=mlp_dims,
            dropout=dropout,
            use_seq_feature=True,
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

    def forward(
        self,
        numerical_x: Optional[torch.Tensor] = None,
        categorical_x: Optional[torch.Tensor] = None,
        seq: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of DeepFM: FM + Deep components

        Args:
            categorical_x: Categorical features (batch_size, num_categorical)
            numerical_x: Numerical features (batch_size, num_numerical)

        Returns:
            DeepFM predictions (batch_size, 1)
        """
        numerical_x = self.bn_num(numerical_x)

        # Get first-order interactions from parent class
        batch_size = (
            categorical_x.size(0) if categorical_x is not None else numerical_x.size(0)
        )

        # Start with bias term (w₀)
        output = self.bias.expand(batch_size).clone()

        # Add first-order interactions (Σwᵢxᵢ)
        output += self._first_order_interactions(numerical_x, categorical_x)

        # sequence encoding
        seq_emb = self.encoder(seq)  # (batch_size, embed_dim)

        # Add second-order interactions
        output += self._second_order_interactions(numerical_x, categorical_x, seq_emb)

        # Concat deep_component and encoded seq_emb
        output += self._deep_component(numerical_x, categorical_x, seq_emb).squeeze(1)

        return output.unsqueeze(-1)  # (batch_size, 1)
