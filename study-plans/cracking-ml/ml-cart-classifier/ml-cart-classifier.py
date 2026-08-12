import numpy as np

def cart_classify(X_train, y_train, X_test, max_depth=5, min_samples=2):
    """
    Returns: list of predicted class labels for each test point
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)


    def get_gini(y_inp):
        a = np.unique(y_inp, return_counts=True)
        class_wise_cnt = a[1]
        total = np.sum(class_wise_cnt)
        gini = 1-np.sum(
                (class_wise_cnt/total)**2
            )
        # print(f"gini={gini}")
        return gini

    def split_dataset(X, y, feat_index, threshold):
        X_feat = X[:, feat_index]
        left = X_feat <= threshold
        right = ~left
        return X[left], y[left], X[right], y[right]

    def find_best_split(X, y):
        g_s = get_gini(y)
        best_gain_value = -1.0
        best_feature, best_threshold = None, None
        n, d = X.shape

        for feat_index in range(d):    
            # unique_vals = np.unique(np.sort(X[:, feat_index])) # np unique always return values in sorted order, so we can avoid sorting here
            # midpoints = (unique_vals[:-1] + unique_vals[1:]) / 2
            thresholds = np.unique(X[:, feat_index])
            for threshold in thresholds.tolist():
                X_left, y_left, X_right, y_right = split_dataset(X, y, feat_index, threshold)

                if len(X_left) == 0 or len(X_right) == 0:
                    continue
                g_sl = get_gini(y_left)
                g_sr = get_gini(y_right)

                gain = g_s - (X_left.shape[0]/len(y)*g_sl) - (X_right.shape[0]/len(y)*g_sr)

                if gain > best_gain_value:
                    best_feature, best_threshold = feat_index, threshold
                    best_gain_value = gain

        return  best_feature, best_threshold, best_gain_value

    def build_tree(X, y, depth):
        if depth >= max_depth or len(y) < min_samples or len(np.unique(y)) == 1:
            classes, counts = np.unique(y, return_counts=True)
            return {
                "leaf": True,
                "label": classes[np.argmax(counts)]
            }

        best_feature, best_threshold, best_gain_value = find_best_split(X, y)
        if best_feature is None or best_gain_value <= 0:
            classes, counts = np.unique(y, return_counts=True)
            return {
                "leaf": True,
                "label": classes[np.argmax(counts)]
            }

        left_mask = X[:, best_feature] <= best_threshold
        X_left = X[left_mask]
        y_left = y[left_mask]
        
        X_right = X[~left_mask]
        y_right = y[~left_mask]
        return {
            "leaf": False,
            "feature": best_feature,
            "thresh": best_threshold,
            "left":  build_tree(X_left, y_left, depth+1),
            "right":  build_tree(X_right, y_right, depth+1)
        }

    def predict_one(node, x):
        if node["leaf"]:
            return node["label"]
        if x[node["feature"]] <= node["thresh"]:
            return predict_one(node["left"], x)
        return predict_one(node["right"], x)

    tree = build_tree(X_train, y_train, 0)
    out = [predict_one(tree, x) for x in X_test]
    return out
