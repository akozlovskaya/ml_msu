import numpy as np

def evaluate_measures(sample):

    p = np.unique(sample, return_counts=True)[1]/len(sample)
    
    h1 = 1 - np.sum(p * p)
    h2 = - np.sum(p * np.log(p))
    h3 = 1 - np.max(p)
    
    measures = {'gini': h1, 'entropy': h2, 'error': h3}
    
    return measures
