import torch

def multi_head_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns: output tensor of shape (batch, seq, d_model)
    """
    q = hidden_states @ w_q
    k = hidden_states @ w_k
    v = hidden_states @ w_v
    
    batch_size, seq_len, d_model = q.shape
    d_k = int(d_model/ num_heads)
    
    q = q.reshape(batch_size, seq_len, num_heads, d_k)
    q = q.transpose(1, 2)
    
    k = k.reshape(batch_size, seq_len, num_heads, d_k)
    k = k.transpose(1, 2)
    
    v = v.reshape(batch_size, seq_len, num_heads, d_k)
    v = v.transpose(1, 2)
    
    scores = q@k.transpose(-2, -1) / d_k**0.5
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, -torch.inf)
    
    attn = torch.softmax(scores, dim=-1)
    output = attn @ v
    
    output = torch.transpose(output, 1, 2)
    output = torch.reshape(output, (batch_size, seq_len, d_model))    
    
    return output@w_o