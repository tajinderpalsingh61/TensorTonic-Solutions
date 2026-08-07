def ridge_regression(X, y, lr, epochs, alpha):
    """
    Perform ridge regression using gradient descent.
    Returns: tuple of (weights_list, bias)
    """
    X = np.array(X)
    y = np.array(y)

    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for i in range(epochs):
        y_pred = X@w + b

        dw = 2/n * X.T@(y_pred-y) + 2*alpha*w
        db = 2/n * (y_pred-y).sum()

        w = w - lr*dw
        b = b - lr*db

    return w, b