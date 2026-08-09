import numpy as np

def knn_classify(X_train, y_train, X_test, k=3):
    """
    Returns: A list of predicted integer labels for each test point
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    tmp = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    tmp **= 2
    euc_dist = np.sqrt(tmp.sum(axis=2))

    nearest_idx = euc_dist.argsort(axis=1)[:, :k]
    neares_labels = y_train[nearest_idx]

    return [np.bincount(row).argmax() for row in neares_labels]
