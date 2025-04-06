import h5py
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from multiprocessing import Pool, cpu_count
from pathlib import Path

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        with Pool(cpu_count()) as pool:
            X_new = pool.map(self.process_signal, X)
        return np.array(X_new)

    def process_signal(self, signal_id):
        with h5py.File(Path('..') / 'data' / 'data.h5', 'r') as hdf:
            for group in hdf.keys():
                signals = hdf[group]
                get_metadata()
                for signal in signals.values():
                    if signal.attrs['id'] == signal_id:
                        get_signal()
                        apply_bp_filter()
                        combine_features()
                        return X_new_row
