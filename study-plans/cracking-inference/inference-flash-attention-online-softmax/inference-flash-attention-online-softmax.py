import torch


def flash_attention_online_softmax(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_block_size: int,
    key_block_size: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns: attention output tensor of shape (batch, seq_q, d_v)
    """
    batch_size, seq_len_q, d_k = query.shape
    _, seq_len_k, _ = key.shape
    _, _, d_v = value.shape

    final_output = torch.zeros((batch_size, seq_len_q, d_v), dtype=query.dtype)

    num_query_blocks = (seq_len_q + query_block_size - 1) // query_block_size
    num_key_blocks = (seq_len_k + key_block_size - 1) // key_block_size

    q_start = 0
    for _ in range(num_query_blocks):
        q_end = min(q_start + query_block_size, seq_len_q)
        actual_q = q_end - q_start

        m = torch.full((batch_size, actual_q), -torch.inf, dtype=query.dtype)
        l = torch.zeros((batch_size, actual_q), dtype=query.dtype)
        o = torch.zeros((batch_size, actual_q, d_v), dtype=query.dtype)

        query_block = query[:, q_start:q_end, :]

        k_start = 0
        for _ in range(num_key_blocks):
            k_end = min(k_start + key_block_size, seq_len_k)

            if causal and k_start >= q_end:
                k_start = k_end
                continue

            key_block = key[:, k_start:k_end, :]
            value_block = value[:, k_start:k_end, :]

            scores = (query_block @ key_block.transpose(-2, -1)) / (d_k ** 0.5)

            if causal:
                query_positions = torch.arange(q_start, q_end)
                key_positions = torch.arange(k_start, k_end)
                mask = key_positions[None, :] > query_positions[:, None]
                scores = scores.masked_fill(mask, -torch.inf)

            block_max = scores.max(dim=-1).values
            m_new = torch.maximum(m, block_max)

            correction = torch.nan_to_num(torch.exp(m - m_new), nan=0.0)
            p = torch.nan_to_num(torch.exp(scores - m_new.unsqueeze(-1)), nan=0.0)

            block_l = p.sum(dim=-1)
            block_o = p @ value_block

            l = l * correction + block_l
            o = o * correction.unsqueeze(-1) + block_o
            m = m_new

            k_start = k_end

        output_block = o / l.unsqueeze(-1).clamp_min(1e-20)
        final_output[:, q_start:q_end, :] = output_block

        q_start = q_end

    return final_output