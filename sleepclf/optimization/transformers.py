import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from signal_features import get_signal_features

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, trial):
        alpha_divs = trial.suggest_int('alpha_divs', 1, 2)
        beta_divs = trial.suggest_int('beta_divs', 1, 4)
        gamma_divs = trial.suggest_int('gamma_divs', 1, 5)

        self.trial = trial
        self.n_divs_map = {
                'theta': 1,
                'alpha': alpha_divs,
                'beta': beta_divs,
                'gamma': gamma_divs
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = []
        for x in X:
            X_new.append(self.extract_features(x))
        return pd.DataFrame(X_new)

    def extract_features(self, x):
        with h5py.File(Path('..') / 'data' / 'data.h5', 'r') as hdf:
            group_name, dataset_name = x.split('x')

            dataset_name = 'd' + dataset_name

            group = hdf[group_name]
            group_features = {key: value for key, value in group.attrs.items()}

            dataset = group[dataset_name]
            signal_features = get_signal_features(dataset[:], self.n_divs_map)

            return group_features | signal_features

    def get_m(self):
        return sum(self.n_divs_map.values()) + 3

class CircularEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, period=1):
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        theta = 2*np.pi*X/self.period
        return np.column_stack([np.cos(theta), np.sin(theta)])

    def inverse_transform(self, X):
        X = np.asarray(X)
        theta = np.arctan2(X[:, 1], X[:, 0])
        return theta*self.period/(2*np.pi)
