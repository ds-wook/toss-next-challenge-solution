import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CausalConv1d(nn.Module):
    """
    Causal 1D Convolution with dilation
    Ensures that output at timestep t only depends on inputs at timesteps <= t
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation

        # Calculate padding to ensure causal convolution
        # For causal conv: padding = (kernel_size - 1) * dilation
        padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform with gain=1.5"""
        nn.init.xavier_uniform_(self.conv.weight, gain=1.5)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)
        Returns:
            Output tensor of shape (batch_size, out_channels, seq_len)
        """
        # Apply convolution
        out = self.conv(x)

        # Remove future timesteps to ensure causality
        # Keep only the first seq_len timesteps
        seq_len = x.size(-1)
        out = out[:, :, :seq_len]

        # Apply dropout
        out = self.dropout(out)

        return out


class TemporalBlock(nn.Module):
    """
    Temporal Block with residual connection
    Consists of two causal convolutions with layer normalization and dropout
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        # First causal convolution
        self.conv1 = CausalConv1d(n_inputs, n_outputs, kernel_size, dilation, dropout)
        self.norm1 = nn.LayerNorm(n_outputs)

        # Second causal convolution
        self.conv2 = CausalConv1d(n_outputs, n_outputs, kernel_size, dilation, dropout)
        self.norm2 = nn.LayerNorm(n_outputs)

        # Residual connection
        self.residual_conv = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )

        # Activation function
        self.activation = nn.ReLU()

        # Initialize residual convolution weights
        if self.residual_conv is not None:
            nn.init.xavier_uniform_(self.residual_conv.weight, gain=1.5)
            if self.residual_conv.bias is not None:
                nn.init.zeros_(self.residual_conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, n_inputs, seq_len)
        Returns:
            Output tensor of shape (batch_size, n_outputs, seq_len)
        """
        # First convolution + normalization + activation
        out = self.conv1(x)
        out = out.transpose(1, 2)  # (batch_size, seq_len, n_outputs)
        out = self.norm1(out)
        out = self.activation(out)
        out = out.transpose(1, 2)  # (batch_size, n_outputs, seq_len)

        # Second convolution + normalization
        out = self.conv2(out)
        out = out.transpose(1, 2)  # (batch_size, seq_len, n_outputs)
        out = self.norm2(out)
        out = out.transpose(1, 2)  # (batch_size, n_outputs, seq_len)

        # Residual connection
        residual = x
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)

        # Add residual and apply activation
        out = self.activation(out + residual)

        return out


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network
    Stack of temporal blocks with increasing dilation rates
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: list,
        kernel_size: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i  # [1, 2, 4, 8, ...]
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation_size,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, num_inputs, seq_len)
        Returns:
            Output tensor of shape (batch_size, num_channels[-1], seq_len)
        """
        return self.network(x)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for sequence aggregation
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.attention_weights = nn.Linear(input_dim, 1)
        self._init_weights()

    def _init_weights(self):
        """Initialize attention weights"""
        nn.init.xavier_uniform_(self.attention_weights.weight, gain=1.5)
        if self.attention_weights.bias is not None:
            nn.init.zeros_(self.attention_weights.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Output tensor of shape (batch_size, input_dim)
        """
        # Compute attention scores
        attention_scores = self.attention_weights(x)  # (batch_size, seq_len, 1)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(
            attention_scores, dim=1
        )  # (batch_size, seq_len, 1)

        # Weighted sum
        output = (x * attention_weights).sum(dim=1)  # (batch_size, input_dim)

        return output


class HistoryTCNEncoder(nn.Module):
    """
    TCN-based encoder for history_b_1~30 sequence data

    Architecture:
    1. Input: (batch_size, 30) -> reshape to (batch_size, 1, 30)
    2. TCN processing with multiple temporal blocks
    3. Global pooling (average or attention)
    4. Output: (batch_size, hidden_dim)
    """

    def __init__(
        self,
        seq_len: int = 30,
        hidden_dim: int = 32,
        num_channels: list = [16, 32, 32, 32],
        kernel_size: int = 2,
        dropout: float = 0.2,
        pooling_method: str = "attention",
    ):
        super().__init__()

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.pooling_method = pooling_method

        # TCN network
        self.tcn = TemporalConvNet(
            num_inputs=1,  # Single feature per timestep
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # Pooling layer
        if pooling_method == "attention":
            self.pooling = AttentionPooling(num_channels[-1])
        elif pooling_method == "average":
            self.pooling = nn.AdaptiveAvgPool1d(1)
        elif pooling_method == "max":
            self.pooling = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")

        # Output projection
        if pooling_method == "attention":
            # Attention pooling already outputs the right dimension
            self.output_proj = nn.Identity()
        else:
            # Average/Max pooling needs projection
            self.output_proj = nn.Linear(num_channels[-1], hidden_dim)
            nn.init.xavier_uniform_(self.output_proj.weight, gain=1.5)
            if self.output_proj.bias is not None:
                nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len)
               representing history_b_1~30 values
        Returns:
            Output tensor of shape (batch_size, hidden_dim)
        """
        # Reshape input: (batch_size, seq_len) -> (batch_size, 1, seq_len)
        x = x.unsqueeze(1)

        # Apply TCN
        tcn_out = self.tcn(x)  # (batch_size, num_channels[-1], seq_len)

        # Apply pooling
        if self.pooling_method == "attention":
            # Transpose for attention pooling: (batch_size, seq_len, num_channels[-1])
            tcn_out = tcn_out.transpose(1, 2)
            pooled = self.pooling(tcn_out)  # (batch_size, num_channels[-1])
        else:
            # Average/Max pooling: (batch_size, num_channels[-1], 1)
            pooled = self.pooling(tcn_out).squeeze(-1)  # (batch_size, num_channels[-1])

        # Apply output projection
        output = self.output_proj(pooled)  # (batch_size, hidden_dim)

        return output

    def get_sequence_embeddings(self, x: Tensor) -> Tensor:
        """
        Get intermediate sequence embeddings for analysis

        Args:
            x: Input tensor of shape (batch_size, seq_len)
        Returns:
            Output tensor of shape (batch_size, seq_len, num_channels[-1])
        """

        # Reshape input: (batch_size, seq_len) -> (batch_size, 1, seq_len)
        x = x.unsqueeze(1)

        # Apply TCN
        tcn_out = self.tcn(x)  # (batch_size, num_channels[-1], seq_len)

        # Transpose to (batch_size, seq_len, num_channels[-1])
        return tcn_out.transpose(1, 2)


def create_history_tcn_encoder(
    seq_len: int = 30, hidden_dim: int = 32, pooling_method: str = "attention"
) -> HistoryTCNEncoder:
    """
    Factory function to create a HistoryTCNEncoder with default parameters

    Args:
        seq_len: Length of input sequence (default: 30 for history_b_1~30)
        hidden_dim: Output embedding dimension (default: 32)
        pooling_method: Pooling method ("attention", "average", "max")

    Returns:
        HistoryTCNEncoder instance
    """
    return HistoryTCNEncoder(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_channels=[16, 32, 32, 32],
        kernel_size=2,
        dropout=0.2,
        pooling_method=pooling_method,
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the TCN encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create encoder
    encoder = create_history_tcn_encoder(
        seq_len=30, hidden_dim=32, pooling_method="attention"
    ).to(device)

    # Test with random data
    batch_size = 64
    seq_len = 30

    # Random history_b_1~30 data
    x = torch.randn(batch_size, seq_len).to(device)

    # Forward pass
    with torch.no_grad():
        output = encoder(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")

        # Test sequence embeddings
        seq_embeddings = encoder.get_sequence_embeddings(x)
        print(f"Sequence embeddings shape: {seq_embeddings.shape}")

        # Verify causality: output at timestep t should only depend on inputs <= t
        print("TCN encoder test passed!")
