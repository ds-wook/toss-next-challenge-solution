import torch
import torch.nn as nn
from typing import Optional, List
from models.fm.dcn.dcn_base import DCNBase


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
            use_seq_features=False,
        )

        self.output_layer = nn.Linear(self.final_input_dim, 1)

        self._init_dcn_weights()

    def forward(
        self,
        numerical_x: Optional[torch.Tensor] = None,
        categorical_x: Optional[torch.Tensor] = None,
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

        # Create dense input vector
        dense_input = self._get_all_embeddings(
            numerical_x, categorical_x, is_num_weighted=True
        ).view(batch_size, -1)

        # Cross Network forward
        cross_output = self.cross_network(dense_input)

        # Deep Network forward
        deep_output = self.deep_network(dense_input)

        # Combine cross and deep outputs
        combined_output = torch.cat([cross_output, deep_output], dim=1)

        # Final prediction
        output = self.output_layer(combined_output)

        return output
