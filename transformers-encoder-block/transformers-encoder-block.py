import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    x = np.array(x)
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    
    layer_norm = gamma * ((x - mean) / np.sqrt(np.power(std, 2) + eps)) + beta
    return layer_norm
    

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    B, N, d_model = Q.shape
    num_heads = 4
    d_k = int(d_model/num_heads)
    
    Q = (Q @ W_q).reshape((B, N, num_heads, d_k)).transpose(0, 2, 1, 3)
    K = (K @ W_k).reshape((B, N, num_heads, d_k)).transpose(0, 2, 1, 3)
    V = (V @ W_v).reshape((B, N, num_heads, d_k)).transpose(0, 2, 1, 3)
    
    multi_head_att = softmax((Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)) @ V
    multi_head_att = multi_head_att.reshape(B, N, d_model)
    return multi_head_att

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    hidden = np.dot(x, W1) + b1
    relu = np.maximum(0, hidden)
    hidden_2 = np.dot(relu, W2) + b2
    return hidden_2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    x_dash = layer_norm(x + multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads), gamma1, beta1)
    encoder = layer_norm(x_dash + feed_forward(x_dash, W1, b1, W2, b2), gamma2, beta2)
    return encoder