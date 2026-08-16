import torch

def multi_query_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_query_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns: output tensor of shape (batch, seq, d_model)
    """
    batch_size, seq_len, d_model = hidden_states.shape
    
    q = hidden_states @ w_q
    k = hidden_states @ w_k
    v = hidden_states @ w_v

    d_k = int(d_model / num_query_heads)
    
    q = q.reshape(batch_size, seq_len, num_query_heads, d_k).transpose(1, 2)
    k = k.unsqueeze(1)
    v = v.unsqueeze(1)

    scores = (q @ k.transpose(-2, -1)) / (d_k ** 0.5)

    if causal:
        mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, -torch.inf)

    attn = torch.softmax(scores, dim=-1)
    output = attn @ v
    output = output.transpose(1, 2).reshape(batch_size, seq_len, d_model)
    output = output @ w_o
    return output
