from typing import List

from torch import Tensor

from models.fm.base import Base


class Model(Base):
    """
    Logistic Regression model supporting both categorical and numerical features

    LR formula: ŷ = sigmoid(w₀ + Σwᵢxᵢ)

    Where:
    - w₀ is global bias (intercept)
    - wᵢ are feature weights

    Args:
        categorical_field_dims: List of vocabulary sizes for categorical features
        numerical_field_count: Number of numerical features
    """

    def __init__(
        self,
        categorical_field_dims: List[int] = None,
        numerical_field_count: int = 0,
        **kwargs,
    ):
        super().__init__(
            categorical_field_dims=categorical_field_dims,
            numerical_field_count=numerical_field_count,
        )

    def forward(
        self, numerical_x: Tensor = None, categorical_x: Tensor = None, **kwargs
    ):
        """
        Forward pass of Logistic Regression

        Args:
            categorical_x: Categorical features (batch_size, num_categorical)
            numerical_x: Numerical features (batch_size, num_numerical)

        Returns:
            Logits (batch_size, 1)
        """
        if categorical_x is None and numerical_x is None:
            raise ValueError(
                "At least one of categorical_x or numerical_x must be provided"
            )

        batch_size = (
            categorical_x.size(0) if categorical_x is not None else numerical_x.size(0)
        )

        # Start with bias term (w₀)
        output = self.bias.expand(batch_size).clone()

        # Add first-order interactions (Σwᵢxᵢ)
        output += self._first_order_interactions(
            self.bn_num(numerical_x), categorical_x
        )

        return output.unsqueeze(-1)  # (batch_size, 1)
