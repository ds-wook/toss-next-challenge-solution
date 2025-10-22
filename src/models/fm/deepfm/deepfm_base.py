import torch.nn as nn
from typing import Optional, List
import torch

from models.fm.fm.fm_base import FMBase


class DeepFMBase(FMBase):
    def __init__(
        self,
        categorical_field_dims: Optional[List[int]] = None,
        numerical_field_count: int = 0,
        embed_dim: int = 16,
        mlp_dims: List[int] = [512, 256, 128],
        dropout: float = 0.2,
        use_seq_feature: bool = False,
        **kwargs,
    ):
        # Initialize parent FM class (gets all FM functionality)
        super().__init__(
            categorical_field_dims=categorical_field_dims,
            numerical_field_count=numerical_field_count,
            embed_dim=embed_dim,
        )

        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.use_seq_feature = use_seq_feature

        # Deep component (MLP) - uses same embeddings as FM
        self._setup_deep_component()
        self._init_deep_weights()

    def _setup_deep_component(self):
        """Setup deep neural network component"""
        # Calculate total input dimension for MLP
        # Uses the same embeddings as FM component
        total_embed_dim = self._calculate_total_embed_dim()

        # Build MLP layers
        mlp_layers = []
        input_dim = total_embed_dim

        for hidden_dim in self.mlp_dims:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(self.dropout))  # Use same dropout rate
            input_dim = hidden_dim

        # Final output layer
        mlp_layers.append(nn.Linear(input_dim, 1))

        self.mlp = nn.Sequential(*mlp_layers)

    def _init_deep_weights(self):
        """Initialize deep component weights"""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.5)
                nn.init.zeros_(layer.bias)

    def _calculate_total_embed_dim(self):
        """Calculate total input dimension for MLP"""
        if self.use_seq_feature:
            return (
                self.num_categorical + self.numerical_field_count + 1
            ) * self.embed_dim
        else:
            return (self.num_categorical + self.numerical_field_count) * self.embed_dim

    def _deep_component(
        self,
        numerical_x: Optional[torch.Tensor],
        categorical_x: Optional[torch.Tensor],
        seq_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Deep neural network component

        Args:
            categorical_x: Categorical features
            numerical_x: Numerical features

        Returns:
            Deep component output (batch_size, 1)
        """
        batch_size = (
            categorical_x.size(0) if categorical_x is not None else numerical_x.size(0)
        )

        # concatenate all dense embeddings
        embeddings = self._get_all_embeddings(
            numerical_x, categorical_x, seq_emb=seq_emb
        )
        deep_input = embeddings.view(
            batch_size, -1
        )  # (batch_size, total_features * embed_dim)

        # Pass through deep network
        deep_output = self.mlp(deep_input)  # (batch_size, 1)

        return deep_output
