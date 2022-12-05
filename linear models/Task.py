import numpy as np

class Preprocesser:
    
    def __init__(self):
        pass
    
    def fit(self, X, Y=None):
        pass
    
    def transform(self, X):
        pass
    
    def fit_transform(self, X, Y=None):
        pass
    
    
class MyOneHotEncoder(Preprocesser):
    
    def __init__(self, dtype=np.float64):
        super(Preprocesser).__init__()
        self.dtype = dtype
        
    def fit(self, X, Y=None):
        self.features = np.concatenate(np.array([np.unique(col) for col in X.to_numpy().T], dtype=object))
        self.features_lens = X.nunique().values

    
    def transform(self, X):
        return np.apply_along_axis(lambda x: 1 * np.array(np.repeat(np.array(x), self.features_lens, axis=0) == self.features), 1, X.to_numpy())
    
    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)
    
    def get_params(self, deep=True):
        return {"dtype": self.dtype}
    
    
class SimpleCounterEncoder:
    
    def __init__(self, dtype=np.float64):
        self.dtype=dtype
        
    def fit(self, X, Y):
        self.successes = {}
        self.counters = {}
        successes_array = [{col: [{item: np.mean(Y.to_numpy()[np.where(X[col].to_numpy() == item)])} for item in
                                            np.unique(X[col].to_numpy())]} for col in X.columns.to_numpy()]
        counters_array = [{col: [{item: np.where(X[col].to_numpy() == item)[0].size / Y.size} for item in
                                            np.unique(X[col].to_numpy())]} for col in X.columns.to_numpy()]
        for item in successes_array:
            self.successes.update(item)
        for key in self.successes.keys():
            newItem = {}
            for item in self.successes[key]:
                newItem.update(item)
            self.successes[key] = newItem
        for item in counters_array:
            self.counters.update(item)
        for key in self.counters.keys():
            newItem = {}
            for item in self.counters[key]:
                newItem.update(item)
            self.counters[key] = newItem

    def transform(self, X, a=1e-5, b=1e-5):
        p = []
        for col in X.columns:
           p.append(list(map(
                lambda feature: [self.successes[col][feature], self.counters[col][feature], (self.successes[col][feature] + a)/(self.counters[col][feature] + b)], X[col].to_numpy())))
        return np.concatenate(np.array(p), axis=1)

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)
    
    def get_params(self, deep=True):
        return {"dtype": self.dtype}

    
def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_ : (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_ :], idx[:(n_splits - 1) * n_]

    
class FoldCounters:
    
    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds
        
    def fit(self, X, Y, seed=1):
        self.successes_folds = []
        self.counteres_folds = []
        sp = group_k_fold(size=X.shape[0], n_splits=self.n_folds, seed = seed)
        self.splits = [*sp]
        for split in self.splits:
            successes = {}
            counters = {}
            X_fold = X.iloc[split[1]]
            Y_train = Y[split[1]]
            successes_array = [{col: [{item: np.mean(Y_train[np.where(X_fold[col].to_numpy() == item)])} for item in
                                      np.unique(X_fold[col].to_numpy())]} for col in X.columns.to_numpy()]
            counters_array = [{col: [{item: np.where(X_fold[col].to_numpy() == item)[0].size / Y_train.size} for item in
                                     np.unique(X_fold[col].to_numpy())]} for col in X_fold.columns.to_numpy()]
            for item in successes_array:
                successes.update(item)
            for key in successes.keys():
                newItem = {}
                for item in successes[key]:
                    newItem.update(item)
                successes[key] = newItem
            for item in counters_array:
                counters.update(item)
            for key in counters.keys():
                newItem = {}
                for item in counters[key]:
                    newItem.update(item)
                counters[key] = newItem
            self.successes_folds.append(successes)
            self.counteres_folds.append(counters)

    def transform(self, X, a=1e-5, b=1e-5):
        res = np.zeros((X.shape[0], 3*X.shape[1]))
        for i in range(len(self.splits)):
            X_fold = X.iloc[self.splits[i][0]]
            p = []
            for col in X_fold.columns:
                p.append(list(map(
                    lambda feature: [self.successes_folds[i][col][feature], self.counteres_folds[i][col][feature],
                                     (self.successes_folds[i][col][feature] + a) / (self.counteres_folds[i][col][feature] + b)],
                    X_fold[col].to_numpy())))
            ans_t = np.concatenate(np.array(p), axis=1)
            res[self.splits[i][0]] = ans_t

        return res

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)
 
       
def weights(x, y):
    values, counts = np.unique(x,return_counts=True)
    return np.array([np.sum(y[np.where(x == values[i])])/counts[i] for i in range(values.size)])
