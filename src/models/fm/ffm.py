import torch
import torch.nn as nn
from torch import Tensor
from models.fm.lr import Model as LogisticRegression


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
        **kwargs,
    ):
        # Initialize parent class (gets bias + first-order interactions)
        super(Model, self).__init__(categorical_field_dims, numerical_field_count)

        self.embed_dim = embed_dim
        self.total_field_count = self.num_categorical + self.numerical_field_count

        # Add field-aware embeddings for second-order interactions
        if self.num_categorical > 0:
            self._setup_categorical_embeddings()

        if self.numerical_field_count > 0:
            self._setup_numerical_embeddings()

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

    def forward(
        self, numerical_x: Tensor = None, categorical_x: Tensor = None, **kwargs
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

        # Add first-order interactions (Σwᵢxᵢ)
        output += self._first_order_interactions(numerical_x, categorical_x)

        # Add field-aware second-order interactions
        output += self._field_aware_interactions(numerical_x, categorical_x)

        return output.unsqueeze(-1)  # (batch_size, 1)

    def _field_aware_interactions(self, numerical_x: Tensor, categorical_x: Tensor):
        """
        Compute field-aware second-order interactions: ΣᵢΣⱼ>ⱼ⟨vᵢ,fⱼ,vⱼ,fᵢ⟩xᵢxⱼ
        Using torch.einsum for efficient computation.
        """
        batch_size = (
            categorical_x.size(0) if categorical_x is not None else numerical_x.size(0)
        )
        device = next(self.parameters()).device

        # Prepare all feature values
        all_x_values = []

        # Categorical values (always 1 for one-hot)
        if categorical_x is not None and self.num_categorical > 0:
            cat_values = torch.ones(batch_size, self.num_categorical, device=device)
            all_x_values.append(cat_values)

        # Numerical values
        if numerical_x is not None and self.numerical_field_count > 0:
            all_x_values.append(numerical_x)

        if not all_x_values:
            return torch.zeros(batch_size, device=device)

        x_all = torch.cat(all_x_values, dim=1)  # (batch_size, total_fields)

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

    def _get_embedding(
        self,
        feature_idx: Tensor,
        field_idx: Tensor,
        categorical_x: Tensor,
        numerical_x: Tensor,
    ):
        """
        Get embedding for feature_idx when interacting with field_idx

        Args:
            feature_idx: Index of the feature (0 to total_field_count-1)
            field_idx: Index of the field it's interacting with
            categorical_x: Categorical input
            numerical_x: Numerical input

        Returns:
            Embedding tensor (batch_size, embed_dim)
        """
        batch_size = (
            categorical_x.size(0) if categorical_x is not None else numerical_x.size(0)
        )

        # Categorical feature
        if feature_idx < self.num_categorical:
            if categorical_x is not None:
                global_idx = (
                    categorical_x[:, feature_idx]
                    + self.field_offsets_tensor[feature_idx]
                )
                return self.categorical_embeddings[field_idx](global_idx)
            else:
                device = next(self.parameters()).device
                return torch.zeros(batch_size, self.embed_dim, device=device)

        # Numerical feature
        else:
            num_feature_idx = feature_idx - self.num_categorical
            if numerical_x is not None:
                return (
                    self.numerical_embeddings[num_feature_idx, field_idx]
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
            else:
                device = next(self.parameters()).device
                return torch.zeros(batch_size, self.embed_dim, device=device)
                device = next(self.parameters()).device
                return torch.zeros(batch_size, self.embed_dim, device=device)
