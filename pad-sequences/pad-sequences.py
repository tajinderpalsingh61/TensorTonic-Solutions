import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if not max_len:
        max_len = max(len(seq) for seq in seqs)
    
    for i, seq in enumerate(seqs):
        if len(seq) < max_len:
            seq += [pad_value] * (max_len - len(seq))
        else:
            seqs[i] = seq[0: max_len]

    return seqs