from pathlib import Path
from multiprocessing import cpu_count

import numpy as np
import h5py
from memory_profiler import memory_usage
import optuna
from optuna.samplers import NSGAIISampler
from pympler import asizeof
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from transformers import FeatureExtractor, CircularEncoder
from models import DimReductionWrapper, CLFWrapper

k = 5 # number of folds for k-fold CV
n_workers = cpu_count() # number of CPUs to use (set to cpu_count() for HPC)

def load_data():
    with h5py.File(Path('..') / 'data' / 'data.h5', 'r') as hdf:
        X = []
        y = []

        for group in hdf.keys():
            signals = hdf[group]
            for signal in signals.values():
                X.append(signal.attrs['id'])
                y.append(signal.attrs['sleep_stage'])

    X = np.array(X)
    y = np.array(y)

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y

# Split data into training and testing sets
def train_test_split(X, y):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    trainval_idx, test_idx = next(sss.split(X, y))

    X_trainval, y_trainval = X[trainval_idx], y[trainval_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return X_trainval, y_trainval, X_test, y_test

### Define objective/pipeline and run optimization
def objective(trial, cv, n, X_trainval, y_trainval):
    # Feature extraction
    feature_extractor = FeatureExtractor(trial, n_workers)
    m = feature_extractor.get_m()

    # Encoding
    encoder = ColumnTransformer(
        transformers=[
            ('scale', StandardScaler(), ['age']), 
            ('onehot', OneHotEncoder(), ['study', 'sex']),
            ('time', CircularEncoder(), ['lights_off'])
        ]
    )

    # Dimensionality reduction
    dim_reduction = DimReductionWrapper(trial, n, m)
    dim_reduction = dim_reduction.get_dim_reduction()

    # Classification
    clf = CLFWrapper(trial, n, m)
    clf = clf.get_clf()

    # Build pipeline
    pipeline = Pipeline([
        ('feature_extractor', feature_extractor),
        ('encoder', encoder),
        ('dim_reduction', dim_reduction),
        ('clf', clf)
    ])

    # Calculate objectives
    cv_error =  1 - cross_val_score(pipeline, X_train, y_train, cv=cv,
                               scoring='accuracy').mean()
    cv_space = asizeof.asizeof(pipeline)*2**(-20) # MB
    return cv_error, cv_space

# Run optimization
def run_optimization():
    # Load data
    X, y = load_data()

    # Extract test data
    X_trainval, y_trainval, X_test, y_test = train_test_split(X, y)
    
    # Generate folds for k-fold CV
    cv = StratifiedKFold(n_splits=k, shuffle=True)
    n = (k - 1)*(len(X_trainval) // k) # safe lower upper bound for n of a fold

    # Create optuna study
    study = optuna.create_study(sampler=NSGAIISampler(), directions=['minimize', 'minimize'])
    study.optimize(lambda trial: objective(trial, cv, n, X_trainval, y_trainval), n_trials=5)

if __name__ == '__main__':
    run_optimization()
