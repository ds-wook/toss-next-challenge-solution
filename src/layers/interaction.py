from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor


class CIN(nn.Module):
    """
    Compressed Interaction Network (CIN) component for xDeepFM

    Captures explicit vector-wise feature interactions at different orders:
    - Layer 0: X⁰ (original embeddings)
    - Layer 1: X¹ = f(X⁰ ⊙ X⁰) (2nd-order)
    - Layer 2: X² = f(X¹ ⊙ X⁰) (3rd-order)
    - ...
    """

    def __init__(self, num_fields: List[int], cin_layer_dims: int, embed_dim: int):
        super(CIN, self).__init__()

        self.num_fields = num_fields
        self.cin_layer_dims = cin_layer_dims  # [H1, H2, H3, ...]
        self.embed_dim = embed_dim

        # Convolution layers for each CIN layer
        self.conv_layers = nn.ModuleList()

        # Input dimension for first layer is num_fields
        prev_dim = num_fields

        for layer_dim in cin_layer_dims:
            # Each conv layer: (prev_dim * num_fields, layer_dim)
            # This represents the interaction between previous layer and X⁰
            self.conv_layers.append(
                nn.Conv1d(prev_dim * num_fields, layer_dim, kernel_size=1)
            )
            prev_dim = layer_dim

        # Output dimension: sum of all CIN layer dimensions
        self.output_dim = sum(cin_layer_dims)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize CIN convolutional layer weights using Xavier uniform"""
        for conv_layer in self.conv_layers:
            nn.init.xavier_uniform_(conv_layer.weight, gain=1.5)
            if conv_layer.bias is not None:
                nn.init.zeros_(conv_layer.bias)

    def forward(self, embeddings):
        """
        Forward pass of CIN

        Args:
            embeddings: (batch_size, num_fields, embed_dim)

        Returns:
            CIN output: (batch_size, output_dim)
        """
        batch_size, num_fields, embed_dim = embeddings.shape

        # Store outputs from all CIN layers for final concatenation
        cin_outputs = []

        # X⁰: original embeddings
        X0 = embeddings  # (batch_size, num_fields, embed_dim)
        Xk = X0  # Current layer input

        # Process each CIN layer
        for i, conv_layer in enumerate(self.conv_layers):
            # Compute interaction between current layer Xk and original X⁰
            # This creates (k+1)-order interactions

            # Xk: (batch_size, Hk_prev, embed_dim)
            # X0: (batch_size, num_fields, embed_dim)

            # Expand dimensions for broadcasting
            Xk_expanded = Xk.unsqueeze(2)  # (batch_size, Hk_prev, 1, embed_dim)
            X0_expanded = X0.unsqueeze(1)  # (batch_size, 1, num_fields, embed_dim)

            # Element-wise product (Hadamard product)
            interaction = (
                Xk_expanded * X0_expanded
            )  # (batch_size, Hk_prev, num_fields, embed_dim)

            # Reshape for convolution: (batch_size, Hk_prev * num_fields, embed_dim)
            interaction = interaction.view(batch_size, -1, embed_dim)

            # Apply 1D convolution across embedding dimension
            Xk_plus_1 = conv_layer(interaction)  # (batch_size, Hk, embed_dim)

            # Apply activation
            Xk_plus_1 = torch.relu(Xk_plus_1)

            # Sum pooling across embedding dimension to get feature representation
            pooled = torch.sum(Xk_plus_1, dim=2)  # (batch_size, Hk)
            cin_outputs.append(pooled)

            # Update for next iteration
            Xk = Xk_plus_1

        # Concatenate outputs from all CIN layers
        if cin_outputs:
            cin_output = torch.cat(
                cin_outputs, dim=1
            )  # (batch_size, sum(cin_layer_dims))
        else:
            cin_output = torch.zeros(batch_size, 0, device=embeddings.device)

        return cin_output


class CrossNetwork(nn.Module):
    def __init__(self, input_dim: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, 1, bias=True) for _ in range(num_layers)]
        )

    def forward(self, x0: Tensor) -> Tensor:
        x = x0
        for w in self.layers:
            x = x0 * w(x) + x
        return x


class CrossNetworkV2(nn.Module):
    """
    Cross Network V2 with low-rank matrix decomposition

    Improves upon CrossNetwork with:
    - Matrix-based feature crossing instead of element-wise
    - Low-rank decomposition for efficiency
    - Better gradient flow and model capacity
    """

    def __init__(
        self, input_dim: int, num_layers: int, low_rank: int = 32, num_experts: int = 1
    ):
        super(CrossNetworkV2, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.low_rank = low_rank
        self.num_experts = num_experts

        # Create cross layers
        self.cross_layers = nn.ModuleList(
            [CrossLayerV2(input_dim, low_rank, num_experts) for _ in range(num_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through cross network v2

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Output tensor (batch_size, input_dim)
        """
        x0 = x  # Keep original input
        xl = x  # Current layer input

        for cross_layer in self.cross_layers:
            xl = cross_layer(x0, xl)

        return xl


class CrossLayerV2(nn.Module):
    """
    Single cross layer for DCN-v2

    Formula: x_{l+1} = x_0 ⊙ (U_l @ (V_l @ x_l) + C_l @ x_l + b_l) + x_l
    """

    def __init__(self, input_dim: int, low_rank: int, num_experts: int = 1):
        super(CrossLayerV2, self).__init__()
        self.input_dim = input_dim
        self.low_rank = low_rank
        self.num_experts = num_experts

        if num_experts == 1:
            # Single expert (standard DCN-v2)
            self.U = nn.Linear(low_rank, input_dim, bias=False)
            self.V = nn.Linear(input_dim, low_rank, bias=False)
            self.C = nn.Linear(input_dim, input_dim, bias=False)
            self.bias = nn.Parameter(torch.zeros(input_dim))
        else:
            # Multiple experts (MoE style)
            self.experts_U = nn.ModuleList(
                [nn.Linear(low_rank, input_dim, bias=False) for _ in range(num_experts)]
            )
            self.experts_V = nn.ModuleList(
                [nn.Linear(input_dim, low_rank, bias=False) for _ in range(num_experts)]
            )
            self.experts_C = nn.ModuleList(
                [
                    nn.Linear(input_dim, input_dim, bias=False)
                    for _ in range(num_experts)
                ]
            )
            self.gate = nn.Linear(input_dim, num_experts)
            self.bias = nn.Parameter(torch.zeros(input_dim))

        self._init_weights()

    def _init_weights(self):
        """Initialize layer weights"""
        if self.num_experts == 1:
            # Initialize single expert
            nn.init.xavier_normal_(self.U.weight)
            nn.init.xavier_normal_(self.V.weight)
            nn.init.xavier_normal_(self.C.weight)
        else:
            # Initialize multiple experts
            for expert_U, expert_V, expert_C in zip(
                self.experts_U, self.experts_V, self.experts_C
            ):
                nn.init.xavier_normal_(expert_U.weight)
                nn.init.xavier_normal_(expert_V.weight)
                nn.init.xavier_normal_(expert_C.weight)
            nn.init.xavier_normal_(self.gate.weight)

    def forward(self, x0: Tensor, xl: Tensor) -> Tensor:
        """
        Forward pass of cross layer v2

        Args:
            x0: Original input (batch_size, input_dim)
            xl: Current layer input (batch_size, input_dim)

        Returns:
            Next layer output (batch_size, input_dim)
        """
        if self.num_experts == 1:
            # Single expert computation
            # x_{l+1} = x_0 ⊙ (U @ (V @ x_l) + C @ x_l + b) + x_l
            v_output = self.V(xl)  # (batch_size, low_rank)
            u_output = self.U(v_output)  # (batch_size, input_dim)
            c_output = self.C(xl)  # (batch_size, input_dim)

            cross_output = u_output + c_output + self.bias
            result = x0 * cross_output + xl
        else:
            # Multiple experts with gating
            gate_scores = F.softmax(self.gate(xl), dim=-1)  # (batch_size, num_experts)

            expert_outputs = []
            for i, (expert_U, expert_V, expert_C) in enumerate(
                zip(self.experts_U, self.experts_V, self.experts_C)
            ):
                v_output = expert_V(xl)
                u_output = expert_U(v_output)
                c_output = expert_C(xl)
                expert_output = u_output + c_output
                expert_outputs.append(expert_output)

            # Weighted combination of experts
            expert_outputs = torch.stack(
                expert_outputs, dim=-1
            )  # (batch_size, input_dim, num_experts)
            gate_scores = gate_scores.unsqueeze(1)  # (batch_size, 1, num_experts)

            cross_output = torch.sum(expert_outputs * gate_scores, dim=-1) + self.bias
            result = x0 * cross_output + xl

        return result


class SENetBlock(nn.Module):
    """SENET block for feature importance learning"""

    def __init__(self, num_fields: int, reduction_ratio: float = 3):
        super(SENetBlock, self).__init__()
        self.num_fields = num_fields
        self.reduction_dim = max(1, num_fields // reduction_ratio)

        self.fc1 = nn.Linear(num_fields, self.reduction_dim)
        self.fc2 = nn.Linear(self.reduction_dim, num_fields)

    def forward(self, embeddings: Tensor):
        """
        Args:
            embeddings: (batch_size, num_fields, embed_dim)
        Returns:
            reweighted embeddings: (batch_size, num_fields, embed_dim)
        """
        batch_size, num_fields, embed_dim = embeddings.shape

        # Global average pooling across embedding dimension
        pooled = torch.mean(embeddings, dim=2)  # (batch_size, num_fields)

        # Two-layer MLP with ReLU activation
        attention = F.relu(self.fc1(pooled))  # (batch_size, reduction_dim)
        attention = torch.sigmoid(self.fc2(attention))  # (batch_size, num_fields)

        # Reweight embeddings
        attention = attention.unsqueeze(-1)  # (batch_size, num_fields, 1)
        return embeddings * attention


class BilinearInteraction(nn.Module):
    """Bilinear interaction layer for field-wise interactions"""

    def __init__(self, embed_dim: int, num_fields: int):
        super(BilinearInteraction, self).__init__()
        self.embed_dim = embed_dim
        self.num_fields = num_fields

        # Bilinear weight matrix for each field pair
        self.num_interactions = num_fields * (num_fields - 1) // 2
        self.bilinear_weights = nn.Parameter(
            torch.randn(self.num_interactions, embed_dim, embed_dim)
        )

    def forward(self, embeddings: Tensor):
        """
        Args:
            embeddings: (batch_size, num_fields, embed_dim)
        Returns:
            bilinear interactions: (batch_size, num_interactions)
        """
        # batch_size = embeddings.size(0)

        # Generate all field pair indices
        field_indices = torch.triu_indices(self.num_fields, self.num_fields, offset=1)
        i_indices, j_indices = field_indices[0], field_indices[1]

        # Get embeddings for all pairs at once
        vi = embeddings[:, i_indices, :]  # (batch_size, num_interactions, embed_dim)
        vj = embeddings[:, j_indices, :]  # (batch_size, num_interactions, embed_dim)

        # Vectorized bilinear interaction: vi^T * W * vj for all pairs
        # Using einsum for efficient computation
        # 'bke,kef,bkf->bk' means:
        # - 'bke': vi (batch_size, num_interactions, embed_dim)
        # - 'kef': bilinear_weights (num_interactions, embed_dim, embed_dim)
        # - 'bkf': vj (batch_size, num_interactions, embed_dim)
        # - '->bk': output (batch_size, num_interactions)
        # This computes: sum over e,f of vi[b,k,e] * W[k,e,f] * vj[b,k,f]
        # Equivalent to: vi^T @ W @ vj for each interaction k in each batch b
        interactions = torch.einsum("bke,kef,bkf->bk", vi, self.bilinear_weights, vj)

        return interactions  # (batch_size, num_interactions)
