import numpy as nn

def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """

    layer_out =  np.dot(X, W) + b
    return layer_out.tolist()