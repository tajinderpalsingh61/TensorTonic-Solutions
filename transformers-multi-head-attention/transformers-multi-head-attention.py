import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    B, N, d_model = Q.shape
    num_heads = 4
    d_k = int(d_model/num_heads)
    
    Q = (Q @ W_q).reshape((B, N, num_heads, d_k)).transpose(0, 2, 1, 3)
    K = (K @ W_k).reshape((B, N, num_heads, d_k)).transpose(0, 2, 1, 3)
    V = (V @ W_v).reshape((B, N, num_heads, d_k)).transpose(0, 2, 1, 3)
    
    multi_head_att = softmax((Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)) @ V
    multi_head_att = multi_head_att.reshape(B, N, d_model)
    return multi_head_att