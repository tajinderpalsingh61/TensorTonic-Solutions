import torch

def sparse_moe_forward(
    token_states: torch.Tensor,
    router_logits: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """
    Returns: output tensor of shape (num_tokens, d_model)
    """
    num_tokens, d_model = token_states.shape
    total_experts = router_logits.shape[1]
    
    output = torch.zeros((num_tokens, d_model), dtype=token_states.dtype)
    
    sorted_values, sorted_index = router_logits.sort(descending=True, stable=True, dim=-1)
    
    top_k_values = sorted_values[:, :top_k]
    expert_indices = sorted_index[:, :top_k]
    
    routing_weights = torch.softmax(top_k_values, dim=-1)
    routing_weights, expert_indices
    
    for i in range(total_experts):
        token_idx, slot_idx = torch.where(expert_indices == i)
    
        if token_idx.numel() == 0:
            continue
    
        x = token_states[token_idx]
        hidden = torch.relu(x @ w_in[i])
        expert_out = hidden @ w_out[i]
    
        gate = routing_weights[token_idx, slot_idx]
        weighted_out = expert_out * gate.unsqueeze(-1)
    
        output.index_add_(0, token_idx, weighted_out)
    
    return output