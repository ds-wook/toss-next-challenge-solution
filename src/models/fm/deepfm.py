import torch
import torch.nn as nn

from models.fm.fm import Model as FactorizationMachine


class Model(FactorizationMachine):
    """
    DeepFM inheriting from FactorizationMachine

    DeepFM formula: ŷ = FM_output + DNN_output
                     = (w₀ + Σwᵢxᵢ + Σᵢ<ⱼ⟨vᵢ,vⱼ⟩xᵢxⱼ) + DNN(embeddings)

    Inherits FM's bias + first-order + second-order interactions,
    adds deep neural network component for high-order interactions.
    """

    def __init__(
        self,
        categorical_field_dims=None,
        numerical_field_count=0,
        embed_dim=16,
        mlp_dims=[512, 256, 128],
        dropout=0.2,
        **kwargs,
    ):
        # Initialize parent FM class (gets all FM functionality)
        super(Model, self).__init__(
            categorical_field_dims, numerical_field_count, embed_dim
        )

        self.mlp_dims = mlp_dims
        self.dropout = dropout

        # Deep component (MLP) - uses same embeddings as FM
        self._setup_deep_component()
        self._init_deep_weights()

    def _setup_deep_component(self):
        """Setup deep neural network component"""
        # Calculate total input dimension for MLP
        # Uses the same embeddings as FM component
        total_embed_dim = (
            self.num_categorical + self.numerical_field_count
        ) * self.embed_dim

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

    def forward(self, numerical_x=None, categorical_x=None, **kwargs):
        """
        Forward pass of DeepFM: FM + Deep components

        Args:
            categorical_x: Categorical features (batch_size, num_categorical)
            numerical_x: Numerical features (batch_size, num_numerical)

        Returns:
            DeepFM predictions (batch_size, 1)
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

        # Add second-order interactions
        output += self._second_order_interactions(numerical_x, categorical_x)

        # Get deep component output
        output += self._deep_component(numerical_x, categorical_x).squeeze()

        return output.unsqueeze(-1)  # (batch_size, 1)

    def _deep_component(self, numerical_x, categorical_x):
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
        device = next(self.parameters()).device

        # Collect all embeddings (reuse same embeddings as FM!)
        all_embeddings = []

        # Categorical embeddings
        if categorical_x is not None and self.num_categorical > 0:
            global_indices = categorical_x + self.field_offsets_tensor.unsqueeze(0)
            cat_embeddings = self.categorical_embeddings(
                global_indices
            )  # (batch_size, num_categorical, embed_dim)
            all_embeddings.append(cat_embeddings)

        # Numerical embeddings
        if numerical_x is not None and self.numerical_field_count > 0:
            num_embeddings = self.numerical_embeddings.unsqueeze(0).expand(
                batch_size, -1, -1
            )  # (batch_size, num_numerical, embed_dim)
            all_embeddings.append(num_embeddings)

        if not all_embeddings:
            return torch.zeros(batch_size, 1, device=device)

        # Concatenate all embeddings and flatten for MLP input
        embeddings = torch.cat(
            all_embeddings, dim=1
        )  # (batch_size, total_features, embed_dim)
        deep_input = embeddings.view(
            batch_size, -1
        )  # (batch_size, total_features * embed_dim)

        # Pass through deep network
        deep_output = self.mlp(deep_input)  # (batch_size, 1)

        return deep_output
