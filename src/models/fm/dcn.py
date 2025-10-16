import torch
import torch.nn as nn
from models.fm.fm import Model as FactorizationMachine
from layers import CrossNetwork


class Model(FactorizationMachine):
    """
    Deep Cross Network (DCN) model

    Combines cross network for explicit feature crossing with
    deep neural network for implicit feature learning.

    Architecture: Cross Network + Deep Network + Output Layer
    """

    def __init__(
        self,
        categorical_field_dims=None,
        numerical_field_count=0,
        embed_dim=10,
        cross_layers=2,
        deep_layers=[256, 128, 64],
        dropout_rate=0.2,
        **kwargs,
    ):
        # Initialize parent class (FM model)
        super(Model, self).__init__(
            categorical_field_dims=categorical_field_dims,
            numerical_field_count=numerical_field_count,
            embed_dim=embed_dim,
            **kwargs,
        )

        self.cross_layers = cross_layers
        self.deep_layers = deep_layers
        self.dropout_rate = dropout_rate

        # Calculate input dimension for cross and deep networks
        self.input_dim = self._calculate_input_dim()

        # Cross Network
        self.cross_network = CrossNetwork(self.input_dim, cross_layers)

        # Deep Network
        self.deep_network = self._build_deep_network()

        # Final output layer (cross + deep outputs)
        final_input_dim = self.input_dim + (
            deep_layers[-1] if deep_layers else self.input_dim
        )
        self.output_layer = nn.Linear(final_input_dim, 1)

        self._init_dcn_weights()

    def _calculate_input_dim(self):
        """Calculate total input dimension for cross and deep networks"""
        total_dim = 0

        # Categorical features: each field contributes embed_dim
        if self.num_categorical > 0:
            total_dim += self.num_categorical * self.embed_dim

        # Numerical features: each field contributes embed_dim
        if self.numerical_field_count > 0:
            total_dim += self.numerical_field_count * self.embed_dim

        return total_dim

    def _build_deep_network(self):
        """Build deep neural network layers"""
        if not self.deep_layers:
            return nn.Identity()

        layers = []
        input_dim = self.input_dim

        for hidden_dim in self.deep_layers:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                ]
            )
            input_dim = hidden_dim

        return nn.Sequential(*layers)

    def _init_dcn_weights(self):
        """Initialize weights for DCN components"""
        # Initialize output layer
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

        # Initialize deep network layers
        for module in self.deep_network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def _create_dense_input(self, numerical_x, categorical_x):
        """Create dense input vector for cross and deep networks"""
        batch_size = (
            categorical_x.size(0) if categorical_x is not None else numerical_x.size(0)
        )
        device = next(self.parameters()).device

        dense_inputs = []

        # Categorical embeddings
        if categorical_x is not None and self.num_categorical > 0:
            global_indices = categorical_x + self.field_offsets_tensor.unsqueeze(0)
            cat_embeddings = self.categorical_embeddings(global_indices)
            # Flatten categorical embeddings
            cat_dense = cat_embeddings.view(batch_size, -1)
            dense_inputs.append(cat_dense)

        # Numerical embeddings
        if numerical_x is not None and self.numerical_field_count > 0:
            # Element-wise multiply numerical values with embedding vectors
            num_embeddings = self.numerical_embeddings.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            numerical_x_expanded = numerical_x.unsqueeze(-1)
            weighted_num_embeddings = num_embeddings * numerical_x_expanded
            # Flatten numerical embeddings
            num_dense = weighted_num_embeddings.view(batch_size, -1)
            dense_inputs.append(num_dense)

        if not dense_inputs:
            return torch.zeros(batch_size, self.input_dim, device=device)

        return torch.cat(dense_inputs, dim=1)

    def forward(self, numerical_x=None, categorical_x=None, **kwargs):
        """
        Forward pass of DCN model

        Args:
            categorical_x: Categorical features (batch_size, num_categorical)
            numerical_x: Numerical features (batch_size, num_numerical)

        Returns:
            DCN predictions (batch_size, 1)
        """
        numerical_x = self.bn_num(numerical_x)

        # Create dense input vector
        dense_input = self._create_dense_input(numerical_x, categorical_x)

        # Cross Network forward
        cross_output = self.cross_network(dense_input)

        # Deep Network forward
        deep_output = self.deep_network(dense_input)

        # Combine cross and deep outputs
        combined_output = torch.cat([cross_output, deep_output], dim=1)

        # Final prediction
        output = self.output_layer(combined_output)

        return output
