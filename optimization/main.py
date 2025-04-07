import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import optuna
import h5py
from pathlib import Path
from models import DimReductionWrapper, CLFWrapper
from multiprocessing import cpu_count
from transformers import FeatureExtractor, CircularEncoder
from memory_profiler import memory_usage
from optuna.samplers import NSGAIISampler

k = 5 # number of folds for k-fold CV
n_workers = 1 # number of CPUs to use (set to cpu_count() for HPC)

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

    # K-fold cross-validation
    return evaluate_objectives(pipeline, cv, X_trainval, y_trainval)

def evaluate_objectives(pipeline, cv, X_trainval, y_trainval):
    cv_error, cv_memory = np.empty(k), np.empty(k)

    for idx, (train_idx, val_idx) in enumerate(cv.split(X_trainval, y_trainval)):
        # Split data into training and validation sets
        X_train, y_train = X_trainval[train_idx], y_trainval[train_idx]
        X_val, y_val = X_trainval[val_idx], y_trainval[val_idx]
        
        # Calculate objectives for this fold
        error, memory = evaluate_fold_objectives(pipeline, X_train, y_train, X_val, y_val)
        
        cv_error[idx] = error
        cv_memory[idx] = memory

    return np.mean(cv_error), np.mean(cv_memory)

def evaluate_fold_objectives(pipeline, X_train, y_train, X_val, y_val):
    # Train model
    pipeline.fit(X_train, y_train)

    # Calculate objectives
    error = 1 - pipeline.score(X_val, y_val)
    memory = np.max(memory_usage((pipeline.predict, (X_val,))))
    
    return error, memory

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
