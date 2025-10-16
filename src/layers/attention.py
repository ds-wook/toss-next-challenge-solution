import math

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 1000):
        super().__init__()

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_seq_len, d_model]

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerEncoderWithAggregation(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        fm_embedding_dim: int,
        aggregation_method: str = "attention_pool",
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # define aggergation method
        self.aggregator = SequenceAggregator(
            d_model, aggregation_method=aggregation_method
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, fm_embedding_dim)

    def create_padding_mask(self, sequences: Tensor) -> Tensor:
        """Create padding mask for variable length sequences"""
        # sequences: [batch_size, seq_len] with 0s as padding
        return sequences == 0  # True for padding positions

    def forward(self, sequences: Tensor) -> Tensor:
        # sequences: [batch_size, seq_len] - padded sequences with 0s
        batch_size, seq_len = sequences.shape

        # Create padding mask
        padding_mask = self.create_padding_mask(sequences)  # [batch_size, seq_len]

        # Embedding + positional encoding
        embedded = self.embedding(sequences)  # [batch_size, seq_len, d_model]
        embedded = embedded * math.sqrt(self.d_model)  # Scale embeddings
        encoded = self.pos_encoding(embedded)  # Add positional encoding

        # Apply transformer with padding mask
        # Note: src_key_padding_mask expects True for positions to ignore
        transformer_out = self.transformer(
            encoded, src_key_padding_mask=padding_mask
        )  # [batch_size, seq_len, d_model]

        # Aggregate sequence outputs
        aggregated = self.aggregator(
            transformer_out, padding_mask
        )  # [batch_size, d_model]

        # Final projection
        output = self.output_proj(aggregated)  # [batch_size, fm_embedding_dim]

        return output


class MultiHeadAttentionWithAggregation(nn.Module):
    """
    Lightweight Multi-Head Attention encoder - just MHA without full transformer
    Much smaller than TransformerDecoder!
    """

    def __init__(
        self,
        vocab_size,
        fm_embedding_dim: int,
        d_model=32,
        nhead=4,
        max_seq_len=512,
        use_feedforward=False,
        use_causal_mask=False,
        aggregation="attention_pool",
    ):
        super().__init__()
        self.d_model = d_model
        self.use_feedforward = use_feedforward
        self.use_causal_mask = use_causal_mask
        self.aggregation = aggregation

        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Multi-head attention (the core component)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=0.1, batch_first=True
        )

        # Optional: Add feedforward network (makes it slightly heavier but better)
        if use_feedforward:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 2, d_model),
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)

        # define aggergation method
        self.aggregator = SequenceAggregator(d_model, aggregation_method=aggregation)

        # Output projection
        self.output_proj = nn.Linear(d_model, fm_embedding_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize all trainable weights using Xavier uniform with gain=1.5"""
        # Initialize embedding
        nn.init.xavier_uniform_(self.embedding.weight, gain=1.5)

        # Initialize attention weights (MultiheadAttention has multiple internal Linear layers)
        for name, param in self.attention.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param, gain=1.5)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Initialize feedforward network if present
        if self.use_feedforward:
            for module in self.ffn.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=1.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        # Initialize output projection
        nn.init.xavier_uniform_(self.output_proj.weight, gain=1.5)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

        # Note: LayerNorm and SequenceAggregator parameters are left with default initialization

    def create_causal_mask(self, seq_len, device):
        """Create causal mask for attention"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(self, sequences):
        # sequences: [batch_size, seq_len]
        batch_size, seq_len = sequences.shape
        device = sequences.device

        # Create masks
        causal_mask = (
            self.create_causal_mask(seq_len, device) if self.use_causal_mask else None
        )
        padding_mask = sequences == 0  # [batch_size, seq_len]

        # Embedding + positional encoding
        x = self.embedding(sequences)  # [batch_size, seq_len, d_model]
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Multi-head attention with causal mask
        attn_out, _ = self.attention(
            query=x,
            key=x,
            value=x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        # Residual connection + normalization
        if self.use_feedforward:
            x = self.norm1(x + attn_out)
            # Feedforward network
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)
        else:
            x = self.norm(x + attn_out)

        # Aggregate sequence
        aggregated = self.aggregator(x, padding_mask)

        # Final projection
        output = self.output_proj(aggregated)

        return output

    def _aggregate(self, x, padding_mask):
        """Aggregate sequence into single vector"""
        if self.aggregation == "masked_avg":
            # Masked average pooling
            mask_expanded = (~padding_mask).unsqueeze(-1).float()
            masked_out = x * mask_expanded
            seq_lengths = mask_expanded.sum(dim=1)
            return masked_out.sum(dim=1) / (seq_lengths + 1e-8)

        elif self.aggregation == "last_token":
            # Last valid token
            seq_lengths = (~padding_mask).sum(dim=1) - 1
            batch_indices = torch.arange(x.size(0), device=x.device)
            return x[batch_indices, seq_lengths]

        elif self.aggregation == "attention_pool":
            # Learnable attention pooling
            attn_scores = self.attention_pooling(x)  # [batch, seq_len, 1]
            attn_scores = attn_scores.masked_fill(
                padding_mask.unsqueeze(-1), float("-inf")
            )
            attn_weights = F.softmax(attn_scores, dim=1)
            return (x * attn_weights).sum(dim=1)

        elif self.aggregation == "max_pool":
            # Max pooling
            x_masked = x.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))
            return x_masked.max(dim=1)[0]

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


class SequenceAggregator(nn.Module):
    """Different methods to aggregate transformer sequence outputs"""

    def __init__(self, d_model, aggregation_method="masked_avg"):
        super().__init__()
        self.aggregation_method = aggregation_method
        self.d_model = d_model

        # For attention-based pooling
        if aggregation_method == "attention_pool":
            self.attention_weights = nn.Linear(d_model, 1)

        # For hierarchical pooling
        elif aggregation_method == "hierarchical":
            self.local_attention = nn.MultiheadAttention(
                d_model, num_heads=4, batch_first=True
            )
            self.global_proj = nn.Linear(d_model, d_model)

    def masked_average_pooling(self, transformer_out, padding_mask):
        """Original approach - average all positions"""
        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        masked_out = transformer_out * mask_expanded
        seq_lengths = mask_expanded.sum(dim=1)
        return masked_out.sum(dim=1) / (seq_lengths + 1e-8)

    def last_token_pooling(self, transformer_out, padding_mask):
        """Use representation of last valid token"""
        seq_lengths = (~padding_mask).sum(dim=1) - 1  # Last valid index
        batch_indices = torch.arange(transformer_out.size(0))
        return transformer_out[batch_indices, seq_lengths]

    def last_n_average_pooling(self, transformer_out, padding_mask, n=3):
        """Average over last N valid tokens"""
        batch_size, seq_len, d_model = transformer_out.shape
        seq_lengths = (~padding_mask).sum(dim=1)  # Actual lengths

        aggregated = []
        for i in range(batch_size):
            actual_len = seq_lengths[i].item()
            start_idx = max(0, actual_len - n)
            end_idx = actual_len

            if start_idx < end_idx:
                last_n_tokens = transformer_out[i, start_idx:end_idx]
                aggregated.append(last_n_tokens.mean(dim=0))
            else:
                # If sequence too short, use all tokens
                mask_i = ~padding_mask[i]
                if mask_i.any():
                    aggregated.append(transformer_out[i][mask_i].mean(dim=0))
                else:
                    aggregated.append(
                        torch.zeros(d_model, device=transformer_out.device)
                    )

        return torch.stack(aggregated)

    def attention_pooling(self, transformer_out, padding_mask):
        """Learnable attention-based pooling"""
        # Compute attention weights
        attention_scores = self.attention_weights(
            transformer_out
        )  # [batch, seq_len, 1]

        # Mask padding positions
        attention_scores = attention_scores.masked_fill(
            padding_mask.unsqueeze(-1), float("-inf")
        )

        # Softmax to get weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, seq_len, 1]

        # Weighted sum
        return (transformer_out * attention_weights).sum(dim=1)

    def max_pooling(self, transformer_out, padding_mask):
        """Max pooling over sequence dimension"""
        masked_out = transformer_out.masked_fill(
            padding_mask.unsqueeze(-1), float("-inf")
        )
        return masked_out.max(dim=1)[0]

    def forward(self, transformer_out, padding_mask):
        """Apply the specified aggregation method"""
        if self.aggregation_method == "masked_avg":
            return self.masked_average_pooling(transformer_out, padding_mask)
        elif self.aggregation_method == "last_token":
            return self.last_token_pooling(transformer_out, padding_mask)
        elif self.aggregation_method == "last_n_avg":
            return self.last_n_average_pooling(transformer_out, padding_mask, n=3)
        elif self.aggregation_method == "attention_pool":
            return self.attention_pooling(transformer_out, padding_mask)
        elif self.aggregation_method == "max_pool":
            return self.max_pooling(transformer_out, padding_mask)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
