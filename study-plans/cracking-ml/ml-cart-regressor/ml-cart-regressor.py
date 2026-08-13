import numpy as np

def cart_regress(X_train, y_train, X_test, max_depth=5, min_samples=2):
    """
    Returns: list of predicted values rounded to 4 decimal places
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    
    def get_mse(X, y):
        n, d = X.shape
        mse = ((y - y.mean()) ** 2).sum()/n
        return mse
    
    def split_data(X, y, feat_index, threshold):
        X_feat = X[:, feat_index]
        left = X_feat <= threshold
        right = ~left
        return X[left], y[left], X[right], y[right]
    
    def find_best_split(X, y):
        mse = get_mse(X, y)
        best_gain_value = float('-inf')
        best_feature, best_threshold = None, None
        n, d = X.shape
    
        for feat_index in range(d):
            thresholds = np.unique(X[:, feat_index])
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = split_data(X, y, feat_index, threshold)
    
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
    
                gain = mse - (get_mse(X_left, y_left)*len(y_left)/n) - (get_mse(X_right, y_right)*len(y_right)/n)
    
                if gain > best_gain_value:
                    best_feature, best_threshold = feat_index, threshold
                    best_gain_value = gain
    
        return best_feature, best_threshold, best_gain_value
    
    def build_tree(X, y, depth):
        if depth >= max_depth or len(y) < min_samples or len(np.unique(y)) == 1:
            return {
                "leaf": True,
                "value": y.mean()
            }
    
        best_feature, best_threshold, best_gain_value = find_best_split(X, y)
        if best_feature is None or best_gain_value <= 0:
            return {
                "leaf": True,
                "value": y.mean()
            }
    
        X_left, y_left, X_right, y_right = split_data(X, y, best_feature, best_threshold)
    
        return {
            "leaf": False,
            "feature": best_feature,
            "thresh": best_threshold,
            "left":  build_tree(X_left, y_left, depth+1),
            "right":  build_tree(X_right, y_right, depth+1)
        }
    
    def predict_one(node, x):
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["thresh"]:
            return predict_one(node["left"], x)
        return predict_one(node["right"], x)
    
    tree = build_tree(X_train, y_train, 0)
    out = [predict_one(tree, x) for x in X_test]
    return out
