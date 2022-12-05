import numpy as np

class MinMaxScaler:
    def fit(self, data):
        self.min = np.amin(data, axis=0)
        self.max = np.amax(data, axis=0)
        
    def transform(self, data):
        return (data - self.min) / (self.max - self.min)


class StandardScaler:
    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.sd = np.sqrt(np.var(data, axis=0))
        
    def transform(self, data):
        return (data - self.mean) / self.sd