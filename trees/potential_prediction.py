import os

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize
import numpy as np

class PotentialTransformer:

    def fit(self, x, y):
        return self

    def fit_transform(self, x, y):
        return self.transform(x)

    def transform(self, x):
        x_new = []
        for obj in x:
            
            n = len(obj)
            m = len(obj[0])

            max_col = -1
            max_row = -1
            min_col = m + 1
            min_row = n + 1
            
            max_col_zero = -1
            max_row_zero = -1
            min_col_zero = m + 1
            min_row_zero = n + 1
            
            for row in range(n):
                for col in range(m):
                    
                    if obj[row][col] < 20:

                        if min_col > col:
                            min_col = col

                        if min_row > row:
                            min_row = row

                        if max_col < col:
                            max_col = col

                        if max_row < row:
                            max_row = row
                            
                    if obj[row][col] < 0.05:

                        if min_col_zero > col:
                            min_col_zero = col

                        if min_row_zero > row:
                            min_row_zero = row

                        if max_col_zero < col:
                            max_col_zero = col

                        if max_row_zero < row:
                            max_row_zero = row
            
            h_zero = max_row_zero - min_row_zero + 1
            w_zero = max_col_zero - min_col_zero + 1
            new_obj = np.full((n, m), 20.)

            min_col_1_zero = (m - w_zero)//2
            min_row_1_zero = (n - h_zero)//2
            max_col_1_zero = min_col_1_zero + w_zero
            max_row_1_zero = min_row_1_zero + h_zero
            
            for i in range(min_row, max_row + 1):
                for j in range(min_col, max_col + 1):
                    if (min_row_1_zero - min_row_zero + i < n) and (min_col_1_zero - min_col_zero + j < m)\
                    and (min_row_1_zero - min_row_zero + i >= 0) and (min_col_1_zero - min_col_zero + j >= 0): 
                        new_obj[min_row_1_zero - min_row_zero + i][min_col_1_zero - min_col_zero + j] = obj[i][j]
        
            max_col = -1
            max_row = -1
            min_col = m + 1
            min_row = n + 1
            
            for row in range(n):
                for col in range(m):
                    
                    if new_obj[row][col] < 20:

                        if min_col > col:
                            min_col = col

                        if min_row > row:
                            min_row = row

                        if max_col < col:
                            max_col = col

                        if max_row < row:
                            max_row = row
                            
            if min_row < (n - max_row):
                for row in range(n - max_row, n - min_row):
                    new_obj[row] = new_obj[n - 1 - row]
                    
            if min_row > (n - max_row):
                for row in range(n - max_row, min_row):
                    new_obj[row] = new_obj[n - 1 - row]
                    
            new_obj = new_obj.T
            
            if min_col < (m - max_col):
                for col in range(m - max_col, m - min_col):
                    new_obj[col] = new_obj[m - 1 - col]
                    
            if min_col > (m - max_col):
                for col in range(m - max_col, min_col):
                    new_obj[col] = new_obj[m - 1 - col]
                    
            new_obj = new_obj.T
            
            x_new += [new_obj]       
            
        x_new = np.array(x_new)  
        x_new[x_new == 20.] = 2000.       
        return x_new.reshape((x_new.shape[0], -1))

def load_dataset(data_dir):
    files, X, Y = [], [], []
    for file in os.listdir(data_dir):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)

def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    regressor = Pipeline([('vectorizer', PotentialTransformer()), ('extra_trees', ExtraTreesRegressor(
        max_depth = 80, n_estimators = 2000, max_features = 50))])
    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    return {file: value for file, value in zip(test_files, predictions)}


