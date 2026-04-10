import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.array(x)
    
    if x.ndim == 2:                        # (N, F)
        axes = 0
        shape = (1, -1)                    # gamma/beta → (1, F)

    elif x.ndim == 3:                      # (N, L, F)
        axes = (0, 1)
        shape = (1, 1, -1)                 # gamma/beta → (1, 1, F)

    elif x.ndim == 4:                      # (N, C, H, W)
        axes = (0, 2, 3)
        shape = (1, -1, 1, 1)             # gamma/beta → (1, C, 1, 1)

    mean = np.mean(x, axis=axes, keepdims=True)
    var  = np.var(x,  axis=axes, keepdims=True)

    x_cap = (x - mean) / np.sqrt(var + eps)

    gamma = np.array(gamma).reshape(shape)
    beta  = np.array(beta).reshape(shape)

    return gamma * x_cap + beta
    