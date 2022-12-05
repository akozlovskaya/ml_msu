import numpy as np
from collections import defaultdict

def kfold_split(num_objects, num_folds):
    x = np.arange(num_objects)
    k = num_objects//num_folds
    ret = []
    for i in range(num_folds - 1):
        train_idx = np.asarray(x[(x < i*k) | (x >= (i+1)*k)])
        test_idx = np.array(x[(x >= i*k) & (x < (i+1)*k)])
        ret += [(train_idx, test_idx)]
     
    ret += [(np.asarray(x[x < k*(num_folds-1)]), np.asarray(x[x >= k*(num_folds-1)]))]
    return ret
    
    
def knn_cv_score(X, y, parameters, score_function, folds, knn_class):
    dict = {}
    for nb in parameters['n_neighbors']:
        for metr in parameters['metrics']:
            for w in parameters['weights']:
                for norm in parameters['normalizers']:
                    X_scaled = X
                    if norm[0] is not None:
                        scaler = norm[0]
                        scaler.fit(X)
                        X_scaled = scaler.transform(X)
                    val = []
                    for f in folds:
                        train_idx, test_idx = f[0], f[1]
                        X_train, y_train = X_scaled[train_idx], y[train_idx]
                        X_test, y_test = X_scaled[test_idx], y[test_idx]
                        model = knn_class(n_neighbors=nb, weights=w, metric=metr)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        score = score_function(y_test, y_pred)
                        val += [score]
                    dict[(norm[1], nb, metr, w)] = np.mean(np.array(val))
    return dict
 
