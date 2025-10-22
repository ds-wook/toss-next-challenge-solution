import torch
import torch.nn as nn
from typing import Optional, List
from models.fm.dcn.dcn_base import DCNBase
from layers import MultiHeadAttentionWithAggregation


class Model(DCNBase):
    """
    Deep Cross Network (DCN) model

    Combines cross network for explicit feature crossing with
    deep neural network for implicit feature learning.

    Architecture: Cross Network + Deep Network + Output Layer
    """

    def __init__(
        self,
        categorical_field_dims: Optional[List[int]] = None,
        numerical_field_count: int = 0,
        embed_dim: int = 10,
        vocab_size: int = 0,
        d_model: int = 128,
        nhead: int = 8,
        use_causal_mask: bool = False,
        max_seq_length: int = 512,
        cross_layers: int = 2,
        deep_layers: List[int] = [256, 128, 64],
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        # Initialize parent class (FM model)
        super().__init__(
            categorical_field_dims=categorical_field_dims,
            numerical_field_count=numerical_field_count,
            embed_dim=embed_dim,
            cross_layers=cross_layers,
            deep_layers=deep_layers,
            dropout_rate=dropout_rate,
            use_seq_features=True,
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
                nn.Linear(self.final_input_dim, 256),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(64, 1),
            ]
        )

        self._init_dcn_weights()

    def forward(
        self,
        numerical_x: Optional[torch.Tensor] = None,
        categorical_x: Optional[torch.Tensor] = None,
        seq: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of DCN model

        Args:
            categorical_x: Categorical features (batch_size, num_categorical)
            numerical_x: Numerical features (batch_size, num_numerical)

        Returns:
            DCN predictions (batch_size, 1)
        """
        batch_size = (
            categorical_x.size(0) if categorical_x is not None else numerical_x.size(0)
        )

        numerical_x = self.bn_num(numerical_x)

        # sequence encoding
        seq_emb = self.encoder(seq)  # (batch_size, embed_dim)

        # Create dense input vector
        dense_input = self._get_all_embeddings(numerical_x, categorical_x).view(
            batch_size, -1
        )

        # Cross Network forward
        cross_output = self.cross_network(dense_input)

        # Deep Network forward
        deep_output = self.deep_network(dense_input)

        # Combine cross and deep outputs
        combined_output = torch.cat([cross_output, deep_output, seq_emb], dim=1)

        # Final prediction
        output = self.output_layer(combined_output)

        return output
