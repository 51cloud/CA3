import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, query, context):
        # query: (B, T1, D), context: (B, T2, D)
        attn_output, _ = self.attn(query, context, context)
        return self.ln(query + attn_output)  # Residual connection
