import numpy as np

# function to test the error of the given model
def get_error(model, X, y):
    y_bar = model.predict(X)
    error = np.mean(y_bar ^ y)
    return error

# function to test the importance of the given target feature
def get_permutated_error(model, target, X, y):
    for feat in X.labels:
        if feat == target:
            to_shuffle = X.loc[:, feat].copy()
            shuffled = to_shuffle.values
            np.random.shuffle(shuffled)
            X.loc[:, feat] = shuffled

            perm_error = get_error(model, X, y)
            return perm_error

    return "Target feature not found."