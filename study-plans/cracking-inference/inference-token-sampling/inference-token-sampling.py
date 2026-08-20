import torch

def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    uniform_draws: torch.Tensor,
) -> torch.Tensor:
    """
    Returns: sampled token id tensor of shape (batch,), dtype torch.int64
    """
    if temperature == 0:
        return torch.argmax(logits, axis=1)
    
    batch_size, vocab_size = logits.shape
    uniform_draws = uniform_draws.unsqueeze(-1)
    
    normalized_logits = logits / temperature
    # print(normalized_logits)
    
    probs = torch.softmax(normalized_logits, dim=-1)
    # print(probs)
    
    if top_k > 0:
        top_k_values = torch.topk(probs, top_k, dim=1).values
    
        thresh = top_k_values.min(dim=1).values
        # print(f"thresh={thresh}")
        mask = probs < thresh.unsqueeze(-1)
        probs = probs.masked_fill(mask, 0)
        # print(f"probs={probs}")
    
    if top_p < 1:
        sorted_probs_values, sorted_probs_index = probs.sort(dim=-1, descending=True)
        # print(f"sorted sorted_probs_values = {sorted_probs_values}")
        cum_sum = sorted_probs_values.cumsum(dim=-1)
        discard_mask_sorted = (cum_sum - sorted_probs_values) >= top_p
        keep_mask_sorted = ~discard_mask_sorted
    
        keep_mask_natural = torch.zeros_like(keep_mask_sorted)
        keep_mask_natural.scatter_(dim=1, index=sorted_probs_index, src=keep_mask_sorted)
    
        probs = probs.masked_fill(~keep_mask_natural, 0)
    
    probs = probs / probs.sum(dim=-1, keepdim=True)    
    prob_cumsum = probs.cumsum(dim=-1)
    # print(prob_cumsum)
    token_ids = torch.argmax((prob_cumsum > uniform_draws).int(), dim=-1)
    return token_ids