import numpy as np

def kmeans(X, k, max_iters=100, seed=42):
    """
    Returns: tuple of (labels as list[int], centroids as list[list[float]])
    """
    rng = np.random.RandomState(seed)

    X = np.array(X)
    n, d = X.shape

    idx = rng.choice(n, size=k, replace=False)
    centroids = X[idx]
    data_centroids = centroids

    for i in range(max_iters):
        tmp = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        tmp.shape
        tmp**=2
        dist = np.sqrt(tmp.sum(axis=2))
        data_centroids = np.argmin(dist, axis=1)
        centroids = np.array([X[data_centroids == i].mean(axis=0) if np.any(X[data_centroids == i]) else centroids[i] for i in range(k)])

    return data_centroids, np.round(centroids, 4)
