import torch
from typing import Tuple

def per_channel_int8_quantize(
    x: torch.Tensor,
    channel_axis: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: (quantized int8 tensor, per-channel scale tensor, dequantized float tensor)
    """
    dims = [d for d in range(x.dim()) if d != channel_axis]
    raw_scale = x.abs().amax(dim=dims, keepdim=True) / 127
    scale = torch.where(raw_scale == 0, torch.ones_like(raw_scale), raw_scale)
    quantized_x = torch.clamp(torch.round(x/scale), -127, 127).to(torch.int8)
    dequantized_x = quantized_x * scale
    return quantized_x, scale, dequantized_x