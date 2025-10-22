from typing import List

from torch import Tensor

from models.fm.xdeepfm.xdeepfm_base import xDeepFMBase
from layers import MultiHeadAttentionWithAggregation


class Model(xDeepFMBase):
    """
    xDeepFM (Extreme Deep Factorization Machine) inheriting from xDeepFMBase

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
        super().__init__(
            categorical_field_dims=categorical_field_dims,
            numerical_field_count=numerical_field_count,
            embed_dim=embed_dim,
            cin_layer_dims=cin_layer_dims,
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
        numerical_x: Tensor = None,
        categorical_x: Tensor = None,
        seq: Tensor = None,
        **kwargs,
    ):
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
