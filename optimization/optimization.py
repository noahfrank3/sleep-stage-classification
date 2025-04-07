import numpy as np
import h5py
import optuna
from optuna.study import MaxTrialsCallback
from pympler import asizeof
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from transformers import FeatureExtractor, CircularEncoder
from models import DimReductionWrapper, CLFWrapper

# Loads X and Y from the h5 data file
def load_data(data_path):
    # Loop through each signal and label, adding to X and y respectively
    with h5py.File(data_path / 'data.h5', 'r') as hdf:
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

# Wraps train_test_split to enable persistant storage for reusing train/test split
def train_test_split_wrapper(data_path, X, y):
    # Check if data file exists and load indices
    try:
        idx = np.load(data_path / 'train_test_split_idx.npz')
        train_idx = idx['train_idx']
        test_idx = idx['test_idx']
    # Create new train/test split if no data file exists
    except FileNotFoundError:
        train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, shuffle=True, stratify=y)
        np.savez(data_path / 'train_test_split_idx.npz', train_idx=train_idx, test_idx=test_idx)

    # Split data into training/testing sets
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test

# Define objective and pipeline for optimization
def objective(trial, cv, n, X_train, y_train, n_internal_workers):
    # Feature extraction
    feature_extractor = FeatureExtractor(trial, n_internal_workers)
    m = feature_extractor.get_m()

    # Encode features
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

# Run optimization process
def run_optimization(data_path, k, n_trials, n_trial_workers, n_internal_workers, db_url):
    # Load data
    X, y = load_data(data_path)

    # Separate training and testing data
    X_train, X_test, y_train, y_test = train_test_split_wrapper(data_path, X, y)
    
    # Generate folds for k-fold CV
    cv = StratifiedKFold(n_splits=k, shuffle=True)

    # Define safe lower upper bound for n of a fold training set
    n = (k - 1)*(len(X_train) // k)

    # Turn on optuna logger
    optuna.logging.set_verbosity(optuna.logging.INFO)

    # Load optuna study and run optimization
    study = optuna.load_study(study_name='sleep_stage_classification', storage=db_url)
    study.optimize(
            lambda trial: objective(
                trial,
                cv, 
                n, 
                X_train,
                y_train,
                n_internal_workers
            ),
            n_trials=n_trials,
            n_jobs=n_trial_workers,
            callbacks=[MaxTrialsCallback(n_trials)]
    )
