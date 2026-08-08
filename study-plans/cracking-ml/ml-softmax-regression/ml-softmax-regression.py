import numpy as np

def softmax_regression(X, y, n_classes, lr=0.01, n_iters=1000):
    """
    Returns: tuple (weights, bias) where weights is a 2D list (d x K) and bias is a list of length K
    """

    X = np.array(X)
    y = np.array(y)
    k = n_classes

    n, d = X.shape
    w = np.zeros((d, k))
    b = np.zeros(k)

    def softmax(Z):
        Z_exp = np.exp(Z)
        Z_exp_sum = Z_exp.sum(axis=1).reshape(n, 1)
        return Z_exp/Z_exp_sum

    for i in range(n_iters):

        # print(X.shape, w.shape)
        Z = X@w + b
        y_pred = softmax(Z)
        y_onehot = np.eye(k)[y]

        # print(w.shape, X.T.shape, (y_pred-y_onehot).shape)
        dw = 1/n * X.T@(y_pred-y_onehot)
        db = 1/n * (y_pred-y_onehot).sum(axis=0)

        w = w - lr*dw
        b = b - lr*db


    return w, b