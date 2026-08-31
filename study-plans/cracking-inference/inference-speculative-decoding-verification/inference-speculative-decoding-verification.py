import torch
from typing import Tuple


def verify_speculative_tokens(
    draft_token_ids: torch.Tensor,
    draft_distributions: torch.Tensor,
    target_distributions: torch.Tensor,
    uniform_draws: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: (accepted_prefix, accepted_count, next_token)
    """
    K = draft_token_ids.shape[0]
    accepted_count = 0

    for i in range(K):
        token = draft_token_ids[i].item()
        p_draft = draft_distributions[i, token].item()
        p_target = target_distributions[i, token].item()

        accept_prob = min(1.0, p_target / p_draft) if p_draft > 0 else 0.0

        if uniform_draws[i].item() < accept_prob:
            accepted_count += 1
        else:
            break

    if accepted_count == K:
        next_dist = target_distributions[K]
    else:
        residual = torch.clamp(
            target_distributions[accepted_count] - draft_distributions[accepted_count], min=0.0
        )
        next_dist = residual / residual.sum()

    cumulative = torch.cumsum(next_dist, dim=-1)
    draw = uniform_draws[K].unsqueeze(0)
    next_token = torch.searchsorted(cumulative, draw, right=True).squeeze(0)
    next_token = torch.clamp(next_token, max=next_dist.shape[0] - 1)

    accepted_prefix = draft_token_ids[:accepted_count]
    return accepted_prefix, torch.tensor(accepted_count, dtype=torch.int64), next_token
