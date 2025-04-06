# Sleep waves
# Delta: 0-3.5 Hz
# Theta: 4-7.5 Hz
# Alpha: 8-13 Hz
# Beta: 14-30 Hz
# Gamma 30-100 Hz

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import optuna
import h5py
from pathlib import Path
from models import *
from feature_extractor import FeatureExtractor

'''This module assumes that the data is already divided into training, validation and testing data. The
module can be divided into 4 steps:
1. Bandpass filtering
2. Data regularization 
3. Dimensionality reduction
4. Machine learning model

Each of the possible combinations of the strategies are then addressed in individual sklearn pipelines.'''

k = 5 # number of folds for k-fold CV

# Split data into training and testing sets
def train_test_split():
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

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    train_idx, test_idx = next(sss.split(X, y))

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return X_train, y_train, X_test, y_test

### Define objective/pipeline and run optimization
def objective(trial):
    # Feature extraction and bandpass filter
    alpha_divs = trial.suggest_int('alpha_divs', 1, 2)
    beta_divs = trial.suggest_int('beta_divs', 1, 4)
    gamma_divs = trial.suggest_int('gamma_divs', 1, 5)
    feature_extractor = FeatureExtractor(alpha_divs, beta_divs, gamma_divs)

    # Normalization
    scalar = StandardScaler()

    # Dimensionality reduction
    dim_reduction = trial.suggest_categorical(
            'dim_reduction',
            list(dim_reduction_mappings.keys()))

    if dim_reduction is None:
        dim_reduction = 'passthrough'
    else:
        dim_reduction = dim_reduction_mappings[dim_reduction](trial)

    # Classification
    clf = trial.suggest_categorical('clf', list(clf_mappings.keys()))
    clf = clf_mappings[clf](trial)

    # Build pipeline
    pipeline = Pipeline([
        ('feature_extractor', feature_extractor),
        ('scalar', scalar),
        ('dim_reduction', dim_reduction),
        ('clf', clf)
    ])

    # K-fold cross-validation
    return 1 - cross_val_score(pipeline, X_train, y_train, cv=cv,
                               scoring='accuracy').mean()

# Run optimization
if __name__ == '__main__':
    # Split data into training and testing sets
    X_train, y_train, X_test, y_test = train_test_split()
    
    # Generate folds for k-fold CV
    cv = StratifiedKFold(shuffle=True)
    n_con = len(X_train) // k # safe lower upper bound for n of a fold

    # Create optuna study
    study = optuna.create_study()
    study.optimize(objective, n_trials=5)
    
    # Implement optimization results

