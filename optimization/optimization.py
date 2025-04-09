from multiprocessing import cpu_count
from pathlib import Path

import h5py
from memory_profiler import memory_usage
import numpy as np
import optuna
from optuna.samplers import NSGAIISampler
from optuna.study import MaxTrialsCallback
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from transformers import FeatureExtractor, CircularEncoder
from models import DimReductionWrapper, CLFWrapper

class Optimizer():
    def __init__(self, hpc_enabled, k, n_trials, f_trial_workers):
        self.k = k
        self.n_trials = n_trials
        self.f_trial_workers = f_trial_workers

        self.data_path = Path('..') / 'data'

        if hpc_enabled:
            n_workers = cpu_count()
            self.n_trial_workers = int(self.f_trial_workers*n_workers)
            self.n_internal_workers = n_workers - self.n_trial_workers
        else:
            self.n_trial_workers = 1
            self.n_internal_workers = 1

    # Loads X and Y from the h5 data file
    def load_data(self):
        # Loop through each signal and label, adding to X and y respectively
        X = []
        y = []

        with h5py.File(self.data_path / 'data.h5', 'r') as hdf:
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

        self.X = X
        self.y = y

        self.n_y = len(np.unique(self.y))

    # Wraps train_test_split to enable persistant storage for reusing train/test split
    def train_test_split_wrapper(self):
        # Check if data file exists and load indices
        try:
            idx = np.load(self.data_path / 'train_test_split_idx.npz')
            train_idx = idx['train_idx']
            test_idx = idx['test_idx']
        # Create new train/test split if no data file exists
        except FileNotFoundError:
            train_idx, test_idx = train_test_split(np.arange(len(self.y)), test_size=0.2, shuffle=True, stratify=self.y)
            np.savez(self.data_path / 'train_test_split_idx.npz', train_idx=train_idx, test_idx=test_idx)

        # Split data into training/testing sets
        self.X_trainval = self.X[train_idx]
        self.y_trainval = self.y[train_idx]

        del self.X
        del self.y

    def k_fold(self):
        self.cv = StratifiedKFold(n_splits=self.k, shuffle=True)
        self.n = (self.k - 1)*(len(self.X_trainval) // self.k)
    
    def configure_optuna(self):
        # Define parameters for objective function
        objective_params = {
            'n_internal_workers': self.n_internal_workers,
            'n': self.n,
            'n_y': self.n_y,
            'k': self.k,
            'cv': self.cv,
            'X_trainval': self.X_trainval,
            'y_trainval': self.y_trainval
        }

        optuna.logging.set_verbosity(optuna.logging.INFO)
        
        storage = optuna.storages.JournalStorage(
                optuna.storages.journal.JournalFileBackend(
                    str(self.data_path / 'optuna_data.log')))

        # Load optuna study and run optimization
        study = optuna.create_study(
                study_name='sleep_stage_classification',
                storage=storage,
                load_if_exists=True,
                sampler=NSGAIISampler(),
                directions=['minimize', 'minimize']
        )

        study.optimize(
                lambda trial: objective(trial, objective_params),
                n_trials=self.n_trials,
                n_jobs=self.n_trial_workers,
                callbacks=[MaxTrialsCallback(self.n_trials)],
        )

    def optimize(self):
        self.load_data()
        self.train_test_split_wrapper()
        self.k_fold()
        self.configure_optuna()

# Define objective and pipeline for optimization
def objective(trial, objective_params):
    # Load global parameters
    n_internal_workers = objective_params['n_internal_workers']
    n = objective_params['n']
    n_y = objective_params['n_y']

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
    dim_reduction = DimReductionWrapper(trial, n, m, n_y)
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
    return evaluate_objectives(pipeline, objective_params)

def evaluate_objectives(pipeline, objective_params):
    # Load global parameters
    k = objective_params['k']
    cv = objective_params['cv']
    X_trainval = objective_params['X_trainval']
    y_trainval = objective_params['y_trainval']

    cv_error, cv_memory = np.empty(k), np.empty(k)

    for idx, (train_idx, val_idx) in enumerate(cv.split(X_trainval, y_trainval)):
        # Split data into training and validation sets
        X_train, y_train = X_trainval[train_idx], y_trainval[train_idx]
        X_val, y_val = X_trainval[val_idx], y_trainval[val_idx]
        
        # Calculate objectives for this fold
        error, memory = evaluate_fold_objectives(pipeline, X_train, y_train, X_val, y_val)
        
        cv_error[idx] = error
        cv_memory[idx] = memory

    return np.mean(cv_error), np.max(cv_memory)

def evaluate_fold_objectives(pipeline, X_train, y_train, X_val, y_val):
    # Train model
    pipeline.fit(X_train, y_train)

    # Calculate objectives
    error = 1 - pipeline.score(X_val, y_val)
    memory = np.max(memory_usage((pipeline.predict, (X_val,))))
    
    return error, memory
