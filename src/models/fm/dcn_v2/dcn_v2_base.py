import torch.nn as nn
from typing import Optional, List
from models.fm.base import Base
from layers import CrossNetworkV2, MultiLayerPerceptron


class DCNv2Base(Base):
    """
    Deep Cross Network V2 (DCN-v2)

    Improvements over DCN-v1:
    - Multiple architecture options (parallel, stacked, stacked_parallel)
    - Enhanced cross network with low-rank matrix decomposition
    - Better model capacity and gradient flow
    - Optional mixture of experts in cross layers
    """

    def __init__(
        self,
        categorical_field_dims: Optional[List[int]] = None,
        numerical_field_count: int = 0,
        embed_dim: int = 10,
        cross_layers: int = 2,
        deep_layers: List[int] = [256, 128, 64],
        dropout_rate: float = 0.2,
        structure: str = "stacked",  # 'parallel', 'stacked', 'stacked_parallel'
        use_low_rank: bool = True,
        low_rank: int = 32,
        num_experts: int = 1,
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
        self.structure = structure
        self.use_low_rank = use_low_rank
        self.low_rank = low_rank
        self.num_experts = num_experts
        self.use_seq_features = use_seq_features

        # Add embeddings
        if self.num_categorical > 0:
            self._setup_categorical_embeddings()

        if self.numerical_field_count > 0:
            self._setup_numerical_embeddings()

        # Calculate input dimension for cross and deep networks
        self.input_dim = self._calculate_input_dim()

        self.deep_network = MultiLayerPerceptron(
            input_dim=self.input_dim,
            layers=deep_layers,
            dropout_rate=dropout_rate,
            only_linear_on_last_layer=False,
        )

        # Rebuild networks and output layer for DCN-v2
        self._build_networks()

    def _build_networks(self):
        """Rebuild networks for DCN-v2 architecture"""
        self.cross_network = CrossNetworkV2(
            self.input_dim, self.cross_layers, self.low_rank, self.num_experts
        )

        # Rebuild output layer based on structure
        final_input_dim = self._calculate_final_input_dim()
        self.output_layer = nn.Linear(final_input_dim, 1)

        # For stacked_parallel, we need an additional deep network
        if self.structure == "stacked_parallel":
            self.deep_network_parallel = MultiLayerPerceptron(
                input_dim=self.input_dim,
                layers=self.deep_layers,
                dropout_rate=self.dropout_rate,
                only_linear_on_last_layer=False,
            )

        # Re-initialize weights
        self._init_dcn_weights()

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

    def _calculate_final_input_dim(self) -> int:
        """Calculate final layer input dimension based on structure"""
        deep_output_dim = self.deep_layers[-1] if self.deep_layers else self.input_dim
        deep_output_dim += self.embed_dim if self.use_seq_features else 0

        if self.structure == "parallel":
            # Cross output + Deep output
            return self.input_dim + deep_output_dim
        elif self.structure == "stacked":
            # Only deep output (deep takes cross output as input)
            return deep_output_dim
        elif self.structure == "stacked_parallel":
            # Cross output + Deep(cross) output + Deep(input) output
            return self.input_dim + deep_output_dim + deep_output_dim
        else:
            raise ValueError(f"Unknown structure: {self.structure}")

    def _init_dcn_weights(self):
        """Initialize weights for DCN-v2 components"""
        # Initialize output layer
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

        # Initialize deep network layers
        for module in self.deep_network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

        # Initialize additional deep network for stacked_parallel
        if hasattr(self, "deep_network_parallel"):
            for module in self.deep_network_parallel.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    nn.init.zeros_(module.bias)
