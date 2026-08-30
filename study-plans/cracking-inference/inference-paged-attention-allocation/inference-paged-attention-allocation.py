import torch
from typing import List, Tuple

def allocate_kv_blocks(
    seq_lengths: List[int],
    block_size: int,
    free_block_ids: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: (block_table, blocks_used, remaining_free_blocks)
    """

    seq_lengths = torch.asarray(seq_lengths)
    free_block_ids = torch.asarray(free_block_ids)
    
    max_block_per_seq = int((torch.ceil(torch.max(seq_lengths) / block_size)).item())

    num_seqs = seq_lengths.shape[0]

    blocks_needed_per_seq = [(int(seq_lengths[s].item()) + block_size - 1) // block_size for s in range(num_seqs)]
    total_blocks_needed = sum(blocks_needed_per_seq)
    
    if total_blocks_needed > len(free_block_ids):
        raise RuntimeError(
            f"Not enough free blocks: need {total_blocks_needed}, "
            f"but only {len(free_block_ids)} are available."
        )

    
    block_table = torch.full((num_seqs, max_block_per_seq), -1, dtype=torch.int64)
    blocks_used = torch.zeros(num_seqs, dtype=torch.int64)
    
    i = 0
    for seq in range(num_seqs):
        no_blocks = (seq_lengths[seq] + block_size - 1) // block_size
        blocks_used[seq] = no_blocks
        for slot in range(no_blocks):
            block_table[seq][slot] = free_block_ids[i]
            i += 1
    
    remaining_free_blocks = free_block_ids[i:]

    return (block_table, blocks_used, remaining_free_blocks)