import torch
from typing import Tuple

def cached_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: (outputs, key_cache, value_cache) for the full sequence, built incrementally
    """
    batch_size, seq_len, d_k = query.shape
    d_v = value.shape[-1]

    key_cache = None
    value_cache = None
    outputs = torch.zeros((batch_size, seq_len, d_v), dtype=query.dtype) # V's last dimension (its actual feature width, d_v) is completely free to be anything, since it just determines how wide each output vector ends up being.
    
    for t in range(seq_len):
        q_t = query[:, t: t+1, :]
        k_t = key[:, t: t+1, :]
        v_t = value[:, t: t+1, :]
    
        if key_cache is None:
            key_cache = k_t
            value_cache = v_t
        else:
            key_cache = torch.cat([key_cache, k_t], dim=1)
            value_cache = torch.cat([value_cache, v_t], dim=1)
    
    
        scores_t = (q_t @ key_cache.transpose(-2, -1)) / (d_k ** 0.5)
        attn_t = torch.softmax(scores_t, dim=-1)
        out_t = attn_t @ value_cache
        
        outputs[:, t:t+1, :] = out_t

    return (outputs, key_cache, value_cache)