import torch

def kv_cache_memory_bytes(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_query_heads: int,
    gqa_kv_heads: int,
    head_dim: int,
    mla_latent_dim: int,
    mla_rotary_key_dim: int,
    bytes_per_element: int,
) -> torch.Tensor:
    """
    Returns: torch.int64 tensor of shape (4,) ordered [MHA, MQA, GQA, MLA]
    """
    bytes_mha = batch_size * seq_len * num_layers * 2 * num_query_heads * head_dim * bytes_per_element

    bytes_mqa = batch_size * seq_len * num_layers * 2 * 1 * head_dim * bytes_per_element

    bytes_gqa = batch_size * seq_len * num_layers * 2 * gqa_kv_heads * head_dim * bytes_per_element

    bytes_mla = batch_size * seq_len * num_layers * (mla_latent_dim + mla_rotary_key_dim) * bytes_per_element

    return torch.asarray(
        [
            bytes_mha, bytes_mqa, bytes_gqa, bytes_mla
        ]
    )
