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
        columns = [np.unique(column) for column in X.to_numpy().T]
        self.features = np.concatenate(np.array(columns, dtype=object))
        self.val = X.nunique().values

    
    def transform(self, X):
        func = lambda x: 1 * np.array(np.repeat(np.array(x), self.val, axis=0) == self.features)
        return np.apply_along_axis(func, 1, X.to_numpy())
    
    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)
    
    def get_params(self, deep=True):
        return {"dtype": self.dtype}
    
    
class SimpleCounterEncoder:
    
    def __init__(self, dtype=np.float64):
        self.dtype=dtype
        
    def fit(self, X, Y):
    
        self.suc = {}
        self.cnt = {}
        
        successes = [{x: [{y: np.mean(Y.to_numpy()[np.where(X[x].to_numpy() == y)])} for y in np.unique(X[x].to_numpy())]} for x in X.columns.to_numpy()]
        counters = [{x: [{y: np.where(X[x].to_numpy() == y)[0].size / Y.size} for y in np.unique(X[x].to_numpy())]} for x in X.columns.to_numpy()]
        
        for y in successes:
            self.suc.update(y)
            
        for i in self.suc.keys():
            cur = {}
            for y in self.suc[i]:
                cur.update(y)
            self.suc[i] = cur
            
        for y in counters:
            self.cnt.update(y)
            
        for i in self.cnt.keys():
            cur = {}
            for y in self.cnt[i]:
                cur.update(y)
            self.cnt[i] = cur

    def transform(self, X, a=1e-5, b=1e-5):
        my_list = []
        for y in X.columns:
            my_list.append(list(map(
                lambda x: [self.suc[y][x], self.cnt[y][x], (self.suc[y][x] + a)/(self.cnt[y][x] + b)], X[y].to_numpy())))
        return np.concatenate(np.array(my_list), axis=1)

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
        self.folds_1 = []
        self.folds_2 = []
        sp = group_k_fold(size=X.shape[0], n_splits=self.n_folds, seed = seed)
        self.splits = [*sp]
        for split in self.splits:
            dict_1 = {}
            dict_2 = {}
            X_fold = X.loc[split[1]]
            Y_train = Y[split[1]]
            successes = [{x: [{y: np.mean(Y_train.to_numpy()[np.where(X_fold[x].to_numpy() == y)])} for y in np.unique(X_fold[x].to_numpy())]} for x in X.columns.to_numpy()]
            counters = [{x: [{y: np.where(X_fold[x].to_numpy() == y)[0].size / Y_train.size} for y in np.unique(X_fold[x].to_numpy())]} for x in X_fold.columns.to_numpy()]
            for y in successes:
                dict_1.update(y)
            for i in dict_1.keys():
                cur = {}
                for y in dict_1[i]:
                    cur.update(y)
                dict_1[i] = cur
            for y in counters:
                dict_2.update(y)
            for i in dict_2.keys():
                cur = {}
                for y in dict_2[i]:
                    cur.update(y)
                dict_2[i] = cur
            self.folds_1.append(dict_1)
            self.folds_2.append(dict_2)

    def transform(self, X, a=1e-5, b=1e-5):
        res = np.zeros((X.shape[0], 3*X.shape[1]))
        for i in range(len(self.splits)):
            X_fold = X.loc[self.splits[i][0]]
            mylist = []
            for y in X_fold.columns:
                mylist.append(list(map(lambda x: [self.folds_1[i][y][x], self.folds_2[i][y][x],
                                     (self.folds_1[i][y][x] + a) / (self.folds_2[i][y][x] + b)],
                                      X_fold[y].to_numpy())))
            res[self.splits[i][0]] = np.concatenate(np.array(mylist), axis=1)

        return res

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)
 
       
def weights(x, y):
    values, counts = np.unique(x, return_counts=True)
    return np.array([np.sum(y[np.where(x == values[i])])/counts[i] for i in range(values.size)])
