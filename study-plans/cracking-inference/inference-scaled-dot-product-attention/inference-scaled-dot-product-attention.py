import torch
from typing import Optional

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Returns: attention output tensor of shape (batch, seq_q, d_v)
    """

    batch_size, seq_len, embed_dim = query.shape    
    scores = (query @ key.transpose(-2, -1)) / (embed_dim ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask, -1*torch.inf)
    attn = torch.softmax(scores, dim=-1)
    return attn @ value