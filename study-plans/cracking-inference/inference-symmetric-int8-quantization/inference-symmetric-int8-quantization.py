import torch
from typing import Tuple

def symmetric_int8_quantize(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: (quantized int8 tensor, scale scalar tensor, dequantized float tensor)
    """
    scale = torch.max(torch.abs(x)) / 127

    if scale.item() == 0.0:
        scale = torch.tensor(1.0)
        quantized_x = torch.zeros_like(x, dtype=torch.int8)
    
    quantized_x = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    dequantized_x = quantized_x * scale

    return quantized_x, scale, dequantized_x