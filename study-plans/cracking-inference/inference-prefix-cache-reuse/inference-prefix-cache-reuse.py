import torch
from typing import List, Tuple

def match_prefix_cache(
    request_token_ids: List[int],
    cached_token_blocks: List[List[List[int]]],
    cached_physical_block_ids: List[List[int]],
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: (matched_token_count scalar tensor, reusable_physical_block_ids tensor)
    """
    max_req_block = len(request_token_ids) // block_size
    request_token_blocks = []
    matched_token_count = 0
    reusable_physical_block_ids = []
    
    max_match = 0
    for i in range(max_req_block):
        request_token_blocks.append(request_token_ids[i*block_size: i*block_size+block_size])
    
    max_match = 0
    for c, candidate in enumerate(cached_token_blocks):
        cnt_match = 0
        for i in range(max_req_block):
            if i < len(candidate):
                # print(request_token_blocks[i], candidate[i])
                if request_token_blocks[i] == candidate[i]:
                    cnt_match += 1
                else:
                    break
        if cnt_match > max_match:
            matched_token_count = cnt_match * block_size
            reusable_physical_block_ids = cached_physical_block_ids[c][:cnt_match]
            max_match = cnt_match
    
    
    return torch.asarray(matched_token_count), torch.asarray(reusable_physical_block_ids)
