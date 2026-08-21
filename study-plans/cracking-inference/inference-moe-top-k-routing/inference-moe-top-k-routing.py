import torch
from typing import Tuple

def route_tokens_to_experts(
    router_logits: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: (expert_indices, routing_weights), each of shape (num_tokens, top_k)
    """
    # top_k_values, top_k_index = router_logits.topk(top_k, dim=-1, sorted=True)
    sorted_values, sorted_index = router_logits.sort(descending=True, stable=True, dim=-1)
    top_k_values = sorted_values[:, :top_k]
    top_k_index = sorted_index[:, :top_k]
    
    probs = torch.softmax(top_k_values, dim=-1)
    return top_k_index, probs
