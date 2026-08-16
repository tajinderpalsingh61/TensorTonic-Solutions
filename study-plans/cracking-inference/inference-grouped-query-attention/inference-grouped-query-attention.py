import torch

def grouped_query_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns: output tensor of shape (batch, seq, d_model)
    """
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_query_heads ({num_query_heads}) must be divisible by "
            f"num_kv_heads ({num_kv_heads})"
        )

    
    batch_size, seq_len, d_model = hidden_states.shape

    q = hidden_states @ w_q # ([1, 3, 8])
    k = hidden_states @ w_k # ([1, 3, 4])
    v = hidden_states @ w_v # ([1, 3, 4])
    
    d_k = int(d_model / num_query_heads)
    d_kv = int(k.shape[-1] / num_kv_heads)
    group_size = int(num_query_heads / num_kv_heads)
    
    q = q.reshape(batch_size, seq_len, num_query_heads, d_k).transpose(1, 2) 
    k = k.reshape(batch_size, seq_len, num_kv_heads, d_kv).transpose(1, 2)
    v = v.reshape(batch_size, seq_len, num_kv_heads, d_kv).transpose(1, 2)
    
    k = k.repeat_interleave(group_size, dim=1)
    v = v.repeat_interleave(group_size, dim=1)
    
    scores = q @ k.transpose(-2, -1) / (d_k ** 0.5)
    
    if causal:
        mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, -torch.inf)
    
    attn = torch.softmax(scores, dim=-1)
    output = attn @ v
    output = output.transpose(1, 2).reshape(batch_size, seq_len, d_model)
    output = output @ w_o
    return output