import numpy as np

def dropout(x, p=0.5, rng=None):
    x = np.array(x, dtype=float)
    
    rand_vals = rng.random(x.shape)   # ← use rng, not np.random
    mask = (rand_vals < (1-p)).astype(float) / (1-p)
    out = x * mask
    
    return (out, mask)