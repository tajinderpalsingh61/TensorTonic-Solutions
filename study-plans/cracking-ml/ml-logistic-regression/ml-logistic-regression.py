import numpy as np

def logistic_regression(X, y, lr=0.01, n_iters=1000):
    """
    Returns:
        tuple: (weights, bias) where weights is a list and bias is a float
    """

    X = np.array(X)
    y = np.array(y)

    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    for i in range(n_iters):
        y_pred = sigmoid(X@w + b)

        dw = 1/n * X.T@(y_pred-y)
        db = 1/n * (y_pred-y).sum()

        w = w - lr*dw
        b = b - lr*db

    return (w, b)
        
        
        
