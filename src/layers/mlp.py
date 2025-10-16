import torch
from torch import nn
from torch import Tensor


class FusionNetwork(nn.Module):
    def __init__(self, embed_dim: int, dropout_rate: float = 0.2):
        super().__init__()

        fusion_layers = []
        # emb, seq encoded emb, hadamard product between two
        hidden_dim = embed_dim * 3 // 2
        fusion_layers.append(nn.Linear(embed_dim * 3, hidden_dim))
        fusion_layers.append(nn.ReLU())
        fusion_layers.append(nn.Dropout(dropout_rate))
        fusion_layers.append(nn.Linear(hidden_dim, 1))
        self.fusion = nn.Sequential(*fusion_layers)

    def forward(self, mlp_emb: Tensor, seq_emb: Tensor):
        # fusion layer with dnc embedding and sequence embedding
        hadamard_component = mlp_emb * seq_emb
        combined_component = torch.cat([mlp_emb, hadamard_component, seq_emb], dim=1)
        return self.fusion(combined_component)
