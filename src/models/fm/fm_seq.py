import torch
import torch.nn as nn
from models.fm.lr import Model as LogisticRegression
from layers import MultiHeadAttentionWithAggregation


class Model(LogisticRegression):
    """
    Factorization Machine inheriting from LogisticRegression

    FM formula: ŷ = w₀ + Σwᵢxᵢ + Σᵢ<ⱼ⟨vᵢ,vⱼ⟩xᵢxⱼ

    Inherits first-order interactions from LogisticRegression,
    adds second-order interactions via embeddings.
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
        **kwargs,
    ):
        # Initialize parent class (gets bias + first-order interactions)
        super(Model, self).__init__(categorical_field_dims, numerical_field_count)

        self.embed_dim = embed_dim

        # Add second-order interaction embeddings
        if self.num_categorical > 0:
            self._setup_categorical_embeddings()

        if self.numerical_field_count > 0:
            self._setup_numerical_embeddings()

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

        self._init_embedding_weights()

    def _setup_categorical_embeddings(self):
        """Setup categorical embeddings for second-order interactions"""
        # Reuse the same total_vocab_size from parent class
        total_vocab_size = self.field_offsets[-1]
        self.categorical_embeddings = nn.Embedding(total_vocab_size, self.embed_dim)

    def _setup_numerical_embeddings(self):
        """Setup numerical embeddings for second-order interactions"""
        # Each numerical feature gets its own embed_dim-dimensional latent vector
        self.numerical_embeddings = nn.Parameter(
            torch.randn(self.numerical_field_count, self.embed_dim)
        )

    def _init_embedding_weights(self):
        """Initialize embedding weights for second-order interactions"""
        if hasattr(self, "categorical_embeddings"):
            nn.init.xavier_normal_(self.categorical_embeddings.weight, gain=1.0)

        if hasattr(self, "numerical_embeddings"):
            nn.init.xavier_normal_(self.numerical_embeddings, gain=1.0)

    def forward(self, numerical_x=None, categorical_x=None, seq=None, **kwargs):
        """
        Forward pass of FM: LR + second-order interactions

        Args:
            categorical_x: Categorical features (batch_size, num_categorical)
            numerical_x: Numerical features (batch_size, num_numerical)

        Returns:
            FM predictions (batch_size, 1)
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

        return output.unsqueeze(-1)  # (batch_size, 1)

    def _second_order_interactions(self, numerical_x, categorical_x, seq_emb):
        """
        Compute second-order interactions: Σᵢ<ⱼ⟨vᵢ,vⱼ⟩xᵢxⱼ
        Uses efficient FM formula: 0.5 * (sum_of_squares - square_of_sums)
        """
        batch_size = (
            categorical_x.size(0) if categorical_x is not None else numerical_x.size(0)
        )
        device = next(self.parameters()).device

        # Collect all embeddings and values in tensors (fully vectorized)
        all_embeddings = []
        all_x_values = []

        # Categorical embeddings and values
        if categorical_x is not None and self.num_categorical > 0:
            global_indices = categorical_x + self.field_offsets_tensor.unsqueeze(0)
            cat_embeddings = self.categorical_embeddings(global_indices)
            all_embeddings.append(cat_embeddings)

            # For categorical features, x_i = 1
            cat_x_values = torch.ones(
                batch_size, self.num_categorical, 1, device=device, dtype=torch.float32
            )
            all_x_values.append(cat_x_values)

        # Numerical embeddings and values
        if numerical_x is not None and self.numerical_field_count > 0:
            num_embeddings = self.numerical_embeddings.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            all_embeddings.append(num_embeddings)

            num_x_values = numerical_x.unsqueeze(-1)
            all_x_values.append(num_x_values)

        if not all_embeddings:
            return torch.zeros(batch_size, device=device)

        all_x_values.append(
            torch.ones(batch_size, 1, 1, device=device, dtype=torch.float32)
        )
        all_embeddings.append(seq_emb.unsqueeze(1))  # (batch_size, 1, embed_dim)

        # Concatenate all embeddings and values
        V = torch.cat(all_embeddings, dim=1)  # (batch_size, total_features, embed_dim)
        X = torch.cat(all_x_values, dim=1)  # (batch_size, total_features, 1)

        # Weighted embeddings: vᵢⱼ * xᵢ
        weighted_V = V * X

        # Efficient FM formula
        sum_embeddings = torch.sum(weighted_V, dim=1)  # (batch_size, embed_dim)
        sum_of_squares = torch.sum(sum_embeddings**2, dim=1)  # (batch_size,)

        square_of_embeddings = weighted_V**2
        square_of_sums = torch.sum(square_of_embeddings, dim=(1, 2))  # (batch_size,)

        second_order = 0.5 * (sum_of_squares - square_of_sums)

        # Clamp the final interaction to prevent NaN
        second_order = torch.clamp(second_order, -100, 100)  # Prevent extreme values

        return second_order
