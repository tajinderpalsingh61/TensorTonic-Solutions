import numpy as np

def channel_statistics(batch):
    """
    Returns: dict with keys "mean" and "std", each a list of length C, with every entry rounded to 4 decimals.
    """
    batch = np.array(batch)
    out = {
        "mean": batch.mean(axis=(0, 1, 2)),
        "std": batch.std(axis=(0, 1, 2))
    }
    return out
