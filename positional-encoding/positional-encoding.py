import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """

    a = np.tile(np.arange(d_model), (seq_len, 1))
    a = a//2

    pos = np.tile(np.arange(seq_len).reshape(seq_len, 1), (1, d_model))
    b = pos / (base ** (2*a/d_model))
    # b, b[:, 0::2], b[:, 1::2]
    b[:, 0::2] = np.sin(b[:, 0::2])
    b[:, 1::2] = np.cos(b[:, 1::2])
    return b
    