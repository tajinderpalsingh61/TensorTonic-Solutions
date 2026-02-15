import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    # Write code here
    y_left = []
    y_right = []
    for i in range(len(y)):
        if split_mask[i]:
            y_left.append(y[i])
        else:
            y_right.append(y[i])

    N = len(y)
    Nl = len(y_left)
    Nr = len(y_right)

    entropy = _entropy(y)
    left_entropy = _entropy(y_left)
    right_entropy = _entropy(y_right)

    info_gain = entropy - ((Nl/N*left_entropy) + (Nr/N*right_entropy))
    return info_gain 

def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.
    """

    C = len(X[0])

    best_gain = -1
    best_feature = None
    best_threshold = None

    X = np.array(X)
    for i in range(C):
        values = X[:, i]
        sorted_unq = np.unique(values)
        thresholds = (sorted_unq[:-1] + sorted_unq[1:]) / 2
        for thresh in thresholds:
            split_mask = values < thresh
            if split_mask.all() or (~split_mask).all(): # All True or all False
                continue
            info_gain = information_gain(y, split_mask)
            if info_gain > best_gain:
                best_gain = info_gain
                best_feature = i
                best_threshold = thresh

    return [best_feature, best_threshold.item()]