import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CircularEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, period=1):
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        theta = 2*np.pi*X/self.period
        return np.cos(theta), np.sin(theta)

    def inverse_transform(self, X):
        X = np.asarray(X)
        theta = np.arctan2(X[:, 1], X[:, 0])
        return theta*self.period/(2*np.pi)
