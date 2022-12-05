from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import pandas as pd

import numpy
from numpy import ndarray

def df_processing(df, df_test):
    
    df.drop('genres', axis=1, inplace=True)
    df_test.drop('genres', axis=1, inplace=True)
    df.drop('directors', axis=1, inplace=True)
    df_test.drop('directors', axis=1, inplace=True)
    df.drop('filming_locations', axis=1, inplace=True)
    df_test.drop('filming_locations', axis=1, inplace=True)
    
    for row in df.index:
        if df.loc[row, 'keywords'] == 'unknown':
            df.loc[row, 'keywords'] = [['']]
    keywords = sorted(set(sum(df['keywords'],[])))[1:]
    keywords = [kw for kw in keywords if sum(list(map(lambda kw_list: int(kw in kw_list), df['keywords']))) > 30]
    new_df = pd.DataFrame({kw: list(map(lambda kw_list: int(kw in kw_list), df['keywords'])) for kw in keywords})
    df.drop('keywords', axis=1, inplace=True)
    df = pd.concat([df, new_df], axis = 1)
    
    new_df = pd.DataFrame({kw: list(map(lambda kw_list: int(kw in kw_list), df_test['keywords'])) for kw in keywords})
    df_test.drop('keywords', axis=1, inplace=True)
    df_test = pd.concat([df_test, new_df], axis = 1)

    
    y = df['awards']
    df = df.drop('awards', axis=1)    
    
    cat_names = ['actor_1_gender', 'actor_2_gender', 'actor_0_gender']
    df_cat = df[cat_names]
    df = (pd.concat([df.drop(cat_names, axis=1), pd.get_dummies(df_cat)], axis = 1))
    df.drop(['actor_1_gender_UNKNOWN', 'actor_2_gender_UNKNOWN', 'actor_0_gender_UNKNOWN'], axis=1, inplace = True)
    
    df_cat = df_test[cat_names]
    df_test = (pd.concat([df_test.drop(cat_names, axis=1), pd.get_dummies(df_cat)], axis = 1))
    df_test.drop(['actor_1_gender_UNKNOWN', 'actor_2_gender_UNKNOWN', 'actor_0_gender_UNKNOWN'], axis=1, inplace = True, errors='ignore')
    
    return (df, y, df_test)


def train_model_and_predict(train_file: str, test_file: str) -> ndarray:

    df_train = pd.read_json(train_file, lines=True)
    df_test = pd.read_json(test_file, lines=True)

    X_train, y_train, X_test = df_processing(df_train, df_test)
    parameters = {"n_estimators": 3000, "max_depth": 6, "learning_rate":0.01}

    regressor = XGBRegressor(**parameters)
    regressor.fit(X_train.to_numpy(), y_train.to_numpy())
    y_pred = regressor.predict(X_test.to_numpy())
    
    y_pred = y_pred.astype(numpy.float64, copy=False)
    return y_pred
