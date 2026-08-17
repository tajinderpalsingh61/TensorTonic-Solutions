import torch
from typing import Tuple

def apply_rotary_position_embeddings(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: (rotated query tensor, rotated key tensor), same shapes as inputs
    """
    positions = positions.float()
    batch_size, num_heads, seq_len, d_k = query.shape
    rotation_freqs = int(d_k / 2)
    
    theta_i = torch.arange(rotation_freqs, dtype=torch.float)
    theta_i = 1 / (base ** (2*theta_i / d_k))
    # print(f"theta_i={theta_i}")
    
    rotation_angles = positions[:, torch.newaxis] @ theta_i[torch.newaxis, :]
    # print(f"rotation_angles = {rotation_angles.shape, rotation_angles}")
    
    cos_m_theta = torch.cos(rotation_angles)
    sin_m_theta = torch.sin(rotation_angles)
    # print(f"cos_m_theta, sin_m_theta = {cos_m_theta.shape, sin_m_theta.shape}")
    
    cos_m_theta = cos_m_theta.unsqueeze(0)
    sin_m_theta = sin_m_theta.unsqueeze(0)
    # print(f"cos_m_theta, sin_m_theta = {cos_m_theta.shape, sin_m_theta.shape}")
    
    def calc_rotated(input):
        x_a = input[:, :, :, :d_k:2]
        x_b = input[:, :, :, 1:d_k:2]
        # print(f"x_a, x_b= {x_a.shape, x_b.shape}")
    
        x_a_dash = (x_a * cos_m_theta) - (x_b * sin_m_theta)
        x_b_dash = (x_a * sin_m_theta) + (x_b * cos_m_theta)
        # print(f"x_a_dash, x_b_dash= {x_a_dash.shape, x_b_dash.shape}")
    
        rotated = torch.stack((x_a_dash, x_b_dash), dim=-1).reshape(batch_size, num_heads, seq_len, d_k)
        return rotated
    
    rotated_query = calc_rotated(query)
    rotated_key = calc_rotated(key)

    return rotated_query, rotated_key