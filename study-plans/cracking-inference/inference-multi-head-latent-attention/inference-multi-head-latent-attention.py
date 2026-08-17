import torch
from typing import Tuple

def multi_head_latent_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_down: torch.Tensor,
    w_up_k: torch.Tensor,
    w_up_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: (output tensor of shape (batch, seq, d_model), latent tensor of shape (batch, seq, d_latent))
    """
    q = hidden_states @ w_q
    c = hidden_states @ w_down
    k = c @ w_up_k
    v = c @ w_up_v
    
    # print(f"q={q.shape}")
    # print(f"c={c.shape}")
    # print(f"k={k.shape}")
    # print(f"v={v.shape}")
    
    batch_size, seq_len, d_model = hidden_states.shape
    d_k = int(d_model / num_heads)
    
    q = q.reshape(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    k = k.reshape(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    v = v.reshape(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    
    # print(f"q={q.shape}")
    # print(f"k={k.shape}")
    # print(f"v={v.shape}")
    
    
    scores = (q @ k.transpose(-2, -1)) / (d_k ** 0.5)
    
    if causal:
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool).triu(diagonal=1)
        scores = scores.masked_fill(mask, -torch.inf)
    
    attn = torch.softmax(scores, dim=-1)
    output = attn @ v
    output = output.transpose(1, 2).reshape(batch_size, seq_len, d_model)
    output = output @ w_o
    return output, c