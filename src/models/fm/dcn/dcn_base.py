import torch.nn as nn
from typing import Optional, List
from models.fm.base import Base
from layers import CrossNetwork, MultiLayerPerceptron


class DCNBase(Base):
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
        use_seq_features: bool = False,
        **kwargs,
    ):
        # Initialize parent class (FM model)
        super().__init__(
            categorical_field_dims=categorical_field_dims,
            numerical_field_count=numerical_field_count,
            **kwargs,
        )

        self.embed_dim = embed_dim
        self.cross_layers = cross_layers
        self.deep_layers = deep_layers
        self.dropout_rate = dropout_rate

        # Add embeddings
        if self.num_categorical > 0:
            self._setup_categorical_embeddings()

        if self.numerical_field_count > 0:
            self._setup_numerical_embeddings()

        # Calculate input dimension for cross and deep networks
        self.input_dim = self._calculate_input_dim()

        # Cross Network
        self.cross_network = CrossNetwork(self.input_dim, cross_layers)

        # Deep Network
        self.deep_network = MultiLayerPerceptron(
            input_dim=self.input_dim,
            layers=deep_layers,
            dropout_rate=dropout_rate,
            only_linear_on_last_layer=False,
        )

        # Final output layer (cross + deep outputs)
        self.final_input_dim = (
            self.input_dim
            + (deep_layers[-1] if deep_layers else self.input_dim)
            + (self.embed_dim if use_seq_features else 0)
        )

    def _calculate_input_dim(self) -> int:
        """Calculate total input dimension for cross and deep networks"""
        total_dim = 0

        # Categorical features: each field contributes embed_dim
        if self.num_categorical > 0:
            total_dim += self.num_categorical * self.embed_dim

        # Numerical features: each field contributes embed_dim
        if self.numerical_field_count > 0:
            total_dim += self.numerical_field_count * self.embed_dim

        return total_dim

    def _init_dcn_weights(self):
        """Initialize weights for DCN components"""
        # Initialize output layer
        if isinstance(self.output_layer, nn.Sequential):
            for module in self.output_layer.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    nn.init.zeros_(module.bias)
        else:
            nn.init.xavier_normal_(self.output_layer.weight)
            nn.init.zeros_(self.output_layer.bias)

        # Initialize deep network layers
        for module in self.deep_network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
