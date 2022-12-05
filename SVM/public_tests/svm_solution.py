import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler




def train_svm_and_predict(train_features, train_target, test_features):
    scaler = StandardScaler()
    scaler.fit(train_features)
    X_train_scaled = scaler.transform(train_features)
    X_test_scaled = scaler.transform(test_features)
    model = SVC(kernel='poly', C=3.8, class_weight='balanced', degree=2)
    model.fit(X_train_scaled, train_target)
    result = model.predict(X_test_scaled)
    return result
