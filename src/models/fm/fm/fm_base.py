import torch
from typing import Optional, List

from models.fm.base import Base


class FMBase(Base):
    def __init__(
        self,
        categorical_field_dims: Optional[List[int]] = None,
        numerical_field_count: int = 0,
        embed_dim: int = 10,
        **kwargs,
    ):
        # Initialize parent class (gets bias + first-order interactions)
        super().__init__(categorical_field_dims, numerical_field_count)

        self.embed_dim = embed_dim

        # Add second-order interaction embeddings
        if self.num_categorical > 0:
            self._setup_categorical_embeddings()

        if self.numerical_field_count > 0:
            self._setup_numerical_embeddings()

        self._init_embedding_weights()

    def _second_order_interactions(
        self,
        numerical_x: Optional[torch.Tensor],
        categorical_x: Optional[torch.Tensor],
        seq_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute second-order interactions: Σᵢ<ⱼ⟨vᵢ,vⱼ⟩xᵢxⱼ
        Uses efficient FM formula: 0.5 * (sum_of_squares - square_of_sums)
        """
        # Concatenate all embeddings and values
        V = self._get_all_embeddings(numerical_x, categorical_x, seq_emb)
        use_seq_emb = True if seq_emb is not None else False
        X = self._get_all_x_values(numerical_x, categorical_x, use_seq_emb=use_seq_emb)

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
