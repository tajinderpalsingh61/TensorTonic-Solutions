import numpy as np    

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """

    X = np.array(X)
    y = np.array(y)
    
    n, d = X.shape
    w = np.zeros(d)

    b = 0.0

    for e in range(epochs):
        y_pred = X@w + b
        mse = 1/n * ((y_pred - y) ** 2)


        dw = 2/n * (X.T@(y_pred - y))
        db = 2/n * ((y_pred - y).sum())

        w = w - (lr*dw)
        b = b - (lr*db)

    return (w, b)