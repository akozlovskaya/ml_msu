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
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        self.uniques = np.concatenate(np.array([np.unique(X[f'{column}'].astype(object)) for column in X.columns], dtype=object))
        self.uniques_lens = X.nunique().values
    
    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        transform_string = lambda x: (np.repeat(np.array(x), self.uniques_lens, axis=0) == self.uniques).astype('int')
        return np.apply_along_axis(transform_string, axis=1, arr=X.to_numpy())
    
    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)
    
    def get_params(self, deep=True):
        return {"dtype": self.dtype}
    
class SimpleCounterEncoder:
    
    def __init__(self, dtype=np.float64):
        self.dtype=dtype
        
    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        
        def fit_one_column(Column, Y):
            column_data_list = []
            for value in Column.unique():
                mean = Y[Column==value].mean()
                proportion = Y[Column==value].size / Y.size
                column_data_list.append((value, mean, proportion))

            return column_data_list
        
        self._data = {}
        for column in X.columns:
            column_data_list = fit_one_column(X[f'{column}'], Y)
            self._data[column] = column_data_list

            
    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        
        result_X =  np.empty((X.shape[0],0))
        for column in X.columns:
            df_for_value = np.zeros((X.shape[0], 3))
            for value, mean, proportion in self._data[column]:
                df_for_value[(X[f'{column}']==value).values] = [mean, proportion, (mean + a) / (proportion + b)]
                
            result_X = np.hstack((result_X, df_for_value))
        
        return result_X
            
    
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
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        
        def fit_one_column(Column, Y):
            column_data_list = []
            for value in Column.unique():
                mean = Y[Column==value].mean()
                proportion = Y[Column==value].size / Y.size
                column_data_list.append((value, mean, proportion))

            return column_data_list
        
        self._seed = seed
        self._folds_data = []
        
        for _, other_folds_indexes in group_k_fold(X.shape[0], n_splits=self.n_folds, seed=self._seed):

            X_train = X.loc[other_folds_indexes]
            Y_train = Y.loc[other_folds_indexes]
            fold_info = {}
            
            for column in X_train.columns:
                column_data_list = fit_one_column(X_train[f'{column}'], Y_train)
                fold_info[column] = column_data_list
            
            self._folds_data.append(fold_info)
            
    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        result_X =  np.empty((X.shape[0],0))
        for column in X.columns:
            df_for_value = np.zeros((X.shape[0], 3))
            
            for i, (fold_indexes, _) in enumerate(group_k_fold(X.shape[0], n_splits=self.n_folds, seed=self._seed)):

                mask = np.array([False] * X.shape[0])
                
                print(mask.shape, fold_indexes)
                mask[fold_indexes] = True
                
                fold_data = self._folds_data[i]
                for value, mean, proportion in fold_data[column]:
                    df_for_value[(X[f'{column}']==value).values * mask] = [mean, proportion, (mean + a) / (proportion + b)]

            result_X = np.hstack((result_X, df_for_value))
        
        return result_X
        
    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)
 
       
def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    values, counts = np.unique(x,return_counts=True)
    return np.array([np.sum(y[np.where(x == values[i])])/counts[i] for i in range(values.size)])
