import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pos = np.arange(seq_length)[:, np.newaxis]
    den = np.power(10000, np.arange(0, d_model, 2) / d_model)
    
    pe = np.zeros((seq_length, d_model))
    pe[:, 0::2] = np.sin(pos/den)
    pe[:, 1::2] = np.cos(pos/den)
    return pe