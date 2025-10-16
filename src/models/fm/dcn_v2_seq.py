import torch
import torch.nn as nn
from models.fm.dcn_seq import Model as DCNv1WithSequence
from layers.interaction import CrossNetworkV2


class Model(DCNv1WithSequence):
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
        categorical_field_dims=None,
        numerical_field_count=0,
        embed_dim=10,
        vocab_size: int = 0,
        d_model: int = 128,
        nhead: int = 8,
        use_causal_mask: bool = False,
        max_seq_length: int = 512,
        cross_layers=2,
        deep_layers=[256, 128, 64],
        dropout_rate=0.2,
        structure="stacked",  # 'parallel', 'stacked', 'stacked_parallel'
        use_low_rank=True,
        low_rank=32,
        num_experts=1,
        **kwargs,
    ):
        self.structure = structure
        self.use_low_rank = use_low_rank
        self.low_rank = low_rank
        self.num_experts = num_experts

        # Initialize parent class
        super().__init__(
            categorical_field_dims=categorical_field_dims,
            numerical_field_count=numerical_field_count,
            embed_dim=embed_dim,
            cross_layers=cross_layers,
            deep_layers=deep_layers,
            dropout_rate=dropout_rate,
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            max_seq_length=max_seq_length,
            use_causal_mask=use_causal_mask,
            **kwargs,
        )

        # Rebuild networks and output layer for DCN-v2
        self._rebuild_networks()

    def _rebuild_networks(self):
        """Rebuild networks for DCN-v2 architecture"""
        # Replace cross network with v2 if requested
        if self.use_low_rank:
            self.cross_network = CrossNetworkV2(
                self.input_dim, self.cross_layers, self.low_rank, self.num_experts
            )

        # Rebuild output layer based on structure
        final_input_dim = self._calculate_final_input_dim()
        self.output_layer = nn.Linear(final_input_dim, 1)

        # For stacked_parallel, we need an additional deep network
        if self.structure == "stacked_parallel":
            self.deep_network_parallel = self._build_deep_network()

        # Re-initialize weights
        self._init_dcn_seq_weights()

    def _calculate_final_input_dim(self):
        """Calculate final layer input dimension based on structure"""
        deep_output_dim = self.deep_layers[-1] if self.deep_layers else self.input_dim

        if self.structure == "parallel":
            # Cross output + Deep output + Seq emb
            return self.input_dim + deep_output_dim + self.embed_dim
        elif self.structure == "stacked":
            # Only deep output (deep takes cross output as input) + Seq emb
            return deep_output_dim + self.embed_dim
        elif self.structure == "stacked_parallel":
            # Cross output + Deep(cross) output + Deep(input) output + Seq emb
            return self.input_dim + deep_output_dim + deep_output_dim + self.embed_dim
        else:
            raise ValueError(f"Unknown structure: {self.structure}")

    def _init_dcn_seq_weights(self):
        """Initialize weights for DCN-v2 components"""
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

        # Initialize output layer
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, numerical_x=None, categorical_x=None, seq=None, **kwargs):
        """
        Forward pass of DCN-v2 model

        Args:
            categorical_x: Categorical features (batch_size, num_categorical)
            numerical_x: Numerical features (batch_size, num_numerical)

        Returns:
            DCN-v2 predictions (batch_size, 1)
        """
        numerical_x = self.bn_num(numerical_x)

        # sequence encoding
        seq_emb = self.encoder(seq)  # (batch_size, embed_dim)

        # Create dense input vector
        dense_input = self._create_dense_input(numerical_x, categorical_x)

        if self.structure == "parallel":
            # Original DCN-v1 style (parallel cross and deep)
            cross_output = self.cross_network(dense_input)
            deep_output = self.deep_network(dense_input)
            combined_output = torch.cat([cross_output, deep_output], dim=1)

        elif self.structure == "stacked":
            # Cross network first, then deep network
            cross_output = self.cross_network(dense_input)
            combined_output = self.deep_network(cross_output)

        elif self.structure == "stacked_parallel":
            # Both stacked and parallel paths
            cross_output = self.cross_network(dense_input)
            deep_stacked_output = self.deep_network(cross_output)
            deep_parallel_output = self.deep_network_parallel(dense_input)

            combined_output = torch.cat(
                [cross_output, deep_stacked_output, deep_parallel_output], dim=1
            )

        else:
            raise ValueError(f"Unknown structure: {self.structure}")

        # Combine cross and seq embedding
        combined_output = torch.cat([combined_output, seq_emb], dim=1)

        # Final prediction
        output = self.output_layer(combined_output)
        return output
