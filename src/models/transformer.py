import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import List

# Direct import to avoid dependency issues
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils.layers import CrossNetwork
except ImportError:
    # Fallback: define CrossNetwork locally if import fails
    class CrossNetwork(nn.Module):
        """Cross Network implementation"""

        def __init__(self, input_dim: int, num_layers: int = 3):
            super().__init__()
            self.num_layers = num_layers
            self.w = nn.ModuleList(
                [nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)]
            )
            self.b = nn.ParameterList(
                [nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)]
            )

        def forward(self, x: Tensor) -> Tensor:
            x0 = x
            for i in range(self.num_layers):
                xw = self.w[i](x)
                x = x0 * xw + self.b[i] + x
            return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        # 시퀀스 길이가 최대 길이를 초과하지 않도록 보장
        if seq_len > self.pe.size(0):
            x = x[:, : self.pe.size(0), :]
            seq_len = self.pe.size(0)

        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return self.dropout(x)


class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence features"""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable query vector
        self.query_vector = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            features: [batch, seq_len, d_model]
            mask: [batch, seq_len], True for padding positions
        Returns:
            pooled: [batch, d_model]
        """
        batch_size, seq_len, d_model = features.size()

        # Learnable query vector를 배치 크기에 맞게 확장
        query = self.query_vector.expand(batch_size, -1, -1)  # [batch, 1, d_model]
        query = self.layer_norm(query)

        # Apply attention pooling
        pooled_features, attention_weights = self.attention(
            query=query,
            key=features,
            value=features,
            key_padding_mask=mask,
            need_weights=False,
        )

        # Remove the sequence dimension (batch, 1, d_model) -> (batch, d_model)
        pooled_features = pooled_features.squeeze(1)

        return pooled_features


class TransformerSequenceEncoder(nn.Module):
    """Transformer encoder for sequence features"""

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(1, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output normalization
        self.output_norm = nn.LayerNorm(d_model)

        # Attention Pooling
        self.attention_pooling = AttentionPooling(d_model, dropout)

    def forward(self, seq: Tensor, seq_lengths: Tensor) -> Tensor:
        # seq: [batch, max_len], seq_lengths: [batch]
        batch_size, max_len = seq.size()

        # Input clipping for numerical stability
        seq = torch.clamp(seq, min=-10.0, max=10.0)

        # Input projection
        seq = seq.unsqueeze(-1)  # [batch, max_len, 1]
        seq = self.input_projection(seq)  # [batch, max_len, d_model]

        # Layer normalization
        seq = self.layer_norm(seq)

        # Positional encoding
        seq = self.pos_encoding(seq)

        # Create padding mask
        padding_mask = self.create_padding_mask(seq_lengths, max_len, seq.device)

        # Transformer encoding
        output = self.transformer(seq, src_key_padding_mask=padding_mask)

        # Output normalization
        output = self.output_norm(output)

        # Attention pooling
        output = self.attention_pooling(output, padding_mask)

        return output

    def create_padding_mask(
        self, seq_lengths: Tensor, max_len: int, device: torch.device
    ) -> Tensor:
        # seq_lengths: [batch]
        # return mask: [batch, max_len], True for padding positions
        batch_size = seq_lengths.size(0)
        mask = torch.arange(max_len, device=device).expand(
            batch_size, max_len
        ) >= seq_lengths.unsqueeze(1)
        return mask


class TransformerCTR(nn.Module):
    """Transformer-based CTR prediction model"""

    def __init__(
        self,
        num_features: int,
        cat_cardinalities: List[int],
        emb_dim: int = 16,
        transformer_dim: int = 64,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        hidden_units: List[int] = [512, 256, 128],
        dropout: List[float] = [0.1, 0.2, 0.3],
    ):
        super().__init__()

        # Embedding layers for categorical features
        self.emb_layers = nn.ModuleList(
            [nn.Embedding(cardinality, emb_dim) for cardinality in cat_cardinalities]
        )
        cat_input_dim = emb_dim * len(cat_cardinalities)

        # Numerical features
        self.bn_num = nn.BatchNorm1d(num_features)

        # Transformer for sequence features
        self.transformer_encoder = TransformerSequenceEncoder(
            d_model=transformer_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dropout=dropout[0] if dropout else 0.1,
        )

        # Cross network
        total_dim = num_features + cat_input_dim + transformer_dim
        self.cross = CrossNetwork(total_dim, num_layers=3)

        # Deep network
        input_dim = total_dim
        layers = []
        for i, h in enumerate(hidden_units):
            layers += [
                nn.Linear(input_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout[i % len(dropout)]),
            ]
            input_dim = h

        # Final output layer
        layers += [
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        ]
        self.mlp = nn.Sequential(*layers)

        # seq_feat extraction flag
        self.extract_seq_feat = False
        self.stored_seq_feat = None

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights safely"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(
        self, num_x: Tensor, cat_x: Tensor, seqs: Tensor, seq_lengths: Tensor
    ) -> Tensor:
        # Numerical features
        if num_x.size(1) > 0:
            num_x = self.bn_num(num_x)
        else:
            num_x = torch.empty(
                num_x.size(0), 0, dtype=torch.float32, device=num_x.device
            )

        # Categorical features
        if cat_x.size(1) > 0 and len(self.emb_layers) > 0:
            cat_embs = [emb(cat_x[:, i]) for i, emb in enumerate(self.emb_layers)]
            cat_feat = torch.cat(cat_embs, dim=1)
        else:
            cat_feat = torch.empty(
                num_x.size(0), 0, dtype=torch.float32, device=num_x.device
            )

        # Sequence features with Transformer
        seq_feat = self.transformer_encoder(seqs, seq_lengths)

        # seq_feat extraction mode
        if self.extract_seq_feat:
            self.stored_seq_feat = seq_feat.detach().cpu()

        # Combine all features
        z = torch.cat([num_x, cat_feat, seq_feat], dim=1)

        # Numerical stability clipping
        z = torch.clamp(z, min=-10.0, max=10.0)

        # Cross network
        z_cross = self.cross(z)

        # Numerical stability clipping
        z_cross = torch.clamp(z_cross, min=-10.0, max=10.0)

        # Deep network
        out = self.mlp(z_cross)

        # Final output clipping
        out = torch.clamp(out, min=-20.0, max=20.0)

        return out.squeeze(1)
