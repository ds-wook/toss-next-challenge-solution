import torch
import torch.nn as nn
from torch import Tensor
from models.fm.lr import Model as LogisticRegression
from layers import MultiHeadAttentionWithAggregation


class Model(LogisticRegression):
    """
    Field-aware Factorization Machine inheriting from LogisticRegression

    FFM formula: ŷ = w₀ + Σwᵢxᵢ + ΣᵢΣⱼ>ⱼ⟨vᵢ,fⱼ,vⱼ,fᵢ⟩xᵢxⱼ

    Each feature has separate embeddings for each field it interacts with.
    """

    def __init__(
        self,
        categorical_field_dims=None,
        numerical_field_count=0,
        embed_dim=10,
        vocab_size: int = 0,
        dropout_rate=0.2,
        d_model: int = 128,
        nhead: int = 8,
        use_causal_mask: bool = False,
        max_seq_length: int = 512,
        **kwargs,
    ):
        # Initialize parent class (gets bias + first-order interactions)
        super(Model, self).__init__(
            categorical_field_dims=categorical_field_dims,
            numerical_field_count=numerical_field_count,
        )

        self.embed_dim = embed_dim
        self.total_field_count = self.num_categorical + self.numerical_field_count

        # Add field-aware embeddings for second-order interactions
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

        self.output_layer = nn.Sequential(
            *[
                nn.Linear(embed_dim + 1, 16),  # plus one for first seq embedding
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(16, 1),
            ]
        )

        # Add mlp layer for efficient tensor processing
        self.reduce_mlp = nn.Sequential(
            *[
                nn.Linear(self.total_field_count * embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]
        )

        self._init_embedding_weights()

    def _setup_categorical_embeddings(self):
        """Setup categorical field-aware embeddings"""
        # Each categorical feature has embeddings for each field it can interact with
        total_vocab_size = self.field_offsets[-1]
        self.categorical_embeddings = nn.ModuleList(
            [
                nn.Embedding(total_vocab_size, self.embed_dim)
                for _ in range(self.total_field_count)
            ]
        )

    def _setup_numerical_embeddings(self):
        """Setup numerical field-aware embeddings"""
        # Each numerical feature has embeddings for each field it can interact with
        self.numerical_embeddings = nn.Parameter(
            torch.randn(
                self.numerical_field_count, self.total_field_count, self.embed_dim
            )
        )

    def _init_embedding_weights(self):
        """Initialize field-aware embedding weights"""
        if hasattr(self, "categorical_embeddings"):
            for embedding in self.categorical_embeddings:
                nn.init.xavier_normal_(embedding.weight, gain=1.0)

        if hasattr(self, "numerical_embeddings"):
            nn.init.xavier_normal_(self.numerical_embeddings, gain=1.0)

        for module in self.output_layer.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        numerical_x: Tensor = None,
        categorical_x: Tensor = None,
        seq: Tensor = None,
        **kwargs,
    ):
        """
        Forward pass of FFM: LR + field-aware second-order interactions

        Args:
            categorical_x: Categorical features (batch_size, num_categorical)
            numerical_x: Numerical features (batch_size, num_numerical)

        Returns:
            FFM predictions (batch_size, 1)
        """
        numerical_x = self.bn_num(numerical_x)

        batch_size = (
            categorical_x.size(0) if categorical_x is not None else numerical_x.size(0)
        )

        # Start with bias term (w₀)
        output = self.bias.expand(batch_size).clone()

        # sequence encoding
        seq_emb = self.encoder(seq)  # (batch_size, embed_dim)

        # Add first-order interactions (Σwᵢxᵢ)
        output += self._first_order_interactions(numerical_x, categorical_x)

        batch_size = (
            categorical_x.size(0) if categorical_x is not None else numerical_x.size(0)
        )
        device = next(self.parameters()).device

        # Get all feature values
        x_all = self._get_all_x(batch_size, categorical_x, numerical_x, device)

        # Get all embeddings
        all_embeddings = self._get_all_embeddings(
            batch_size, categorical_x, numerical_x, device
        )

        # Add field-aware second-order interactions
        output += self._field_aware_interactions(x_all, all_embeddings, device)

        # Combine cross and deep outputs
        combined_output = torch.cat([output.unsqueeze(1), seq_emb], dim=1)

        # Final prediction
        output = self.output_layer(combined_output)

        return output

    def _field_aware_interactions(
        self, x_all: Tensor, all_embeddings: Tensor, device: str
    ):
        """
        Compute field-aware second-order interactions: ΣᵢΣⱼ>ⱼ⟨vᵢ,fⱼ,vⱼ,fᵢ⟩xᵢxⱼ
        Using torch.einsum for efficient computation.
        """
        # Create upper triangular mask for i < j interactions
        field_indices = torch.arange(self.total_field_count, device=device)
        i_mask, j_mask = torch.meshgrid(field_indices, field_indices, indexing="ij")
        upper_tri_mask = i_mask < j_mask

        # Use einsum to compute all dot products efficiently
        # 'bijd,bjid->bij' means: for each batch b, compute dot product between
        # embeddings at positions (i,j,d) and (j,i,d) across dimension d
        dot_products = torch.einsum("bijd,bjid->bij", all_embeddings, all_embeddings)

        # Compute all pairwise x_i * x_j products
        x_products = torch.einsum("bi,bj->bij", x_all, x_all)

        # Combine dot products with x products and apply upper triangular mask
        interactions = dot_products * x_products * upper_tri_mask.float()

        # Sum all valid interactions
        interaction_sum = torch.sum(interactions, dim=(1, 2))

        # Clamp to prevent extreme values
        interaction_sum = torch.clamp(interaction_sum, -100, 100)

        return interaction_sum

    def _get_all_x(
        self, batch_size: int, categorical_x: Tensor, numerical_x: Tensor, device: str
    ) -> Tensor:
        """
        Prepare all feature values for field-aware interactions.

        Args:
            batch_size: Batch size
            categorical_x: Categorical features
            numerical_x: Numerical features
            device: Device to place tensors on

        Returns:
            x_all: Concatenated feature values (batch_size, total_fields) or None if no features
        """
        all_x_values = []

        # Categorical values (always 1 for one-hot)
        if categorical_x is not None and self.num_categorical > 0:
            cat_values = torch.ones(batch_size, self.num_categorical, device=device)
            all_x_values.append(cat_values)

        # Numerical values
        if numerical_x is not None and self.numerical_field_count > 0:
            all_x_values.append(numerical_x)

        if not all_x_values:
            return None

        return torch.cat(all_x_values, dim=1)  # (batch_size, total_fields)

    def _get_all_embeddings(
        self,
        batch_size: int,
        categorical_x: Tensor,
        numerical_x: Tensor,
        device,
    ):
        """
        Pre-compute all embeddings for field-aware interactions.

        Args:
            batch_size: Batch size
            categorical_x: Categorical features
            numerical_x: Numerical features
            seq_emb: Sequence embeddings
            device: Device to place tensors on

        Returns:
            all_embeddings: Tensor of shape (batch_size, total_fields, total_fields, embed_dim)
        """
        # Pre-compute all embeddings more efficiently
        # Shape: (batch_size, total_fields, total_fields, embed_dim)
        all_embeddings = torch.zeros(
            batch_size,
            self.total_field_count,
            self.total_field_count,
            self.embed_dim,
            device=device,
        )

        # Fill categorical embeddings - vectorized
        if categorical_x is not None and self.num_categorical > 0:
            global_indices = categorical_x + self.field_offsets_tensor.unsqueeze(0)
            # Stack all categorical embeddings at once
            cat_embeddings_stack = torch.stack(
                [emb(global_indices) for emb in self.categorical_embeddings], dim=2
            )  # (batch_size, num_categorical, total_fields, embed_dim)
            all_embeddings[:, : self.num_categorical, :, :] = cat_embeddings_stack

        # Fill numerical embeddings - vectorized
        if numerical_x is not None and self.numerical_field_count > 0:
            num_start_idx = self.num_categorical
            # Expand numerical embeddings for batch
            num_emb_expanded = self.numerical_embeddings.unsqueeze(0).expand(
                batch_size, -1, -1, -1
            )  # (batch_size, num_numerical, total_fields, embed_dim)
            all_embeddings[:, num_start_idx:, :, :] = num_emb_expanded

        return all_embeddings
