from concurrent.futures import ProcessPoolExecutor
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
import h5py
from memory_profiler import memory_usage
import numpy as np
import optuna
from optuna.samplers import NSGAIISampler
from optuna.storages import RDBStorage
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import unique_labels

from transformers import FeatureExtractor, CircularEncoder
from models import DimReductionWrapper, CLFWrapper

class Optimizer():
    def __init__(self, batch_size, k):
        self.batch_size = batch_size
        self.k = k

        self.data_path = Path('..') / 'data'

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

        logging.info("[Main] Data loaded sucessfully!")

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

        self.n_y = len(unique_labels(self.y_trainval)) - 1

        del self.X
        del self.y

        logging.info("[Main] Train-test-split completed sucessfully!")

    def k_fold(self):
        self.cv = StratifiedKFold(n_splits=self.k, shuffle=True)
        self.n = (self.k - 1)*(len(self.X_trainval) // self.k)

        logging.info("[Main] K-fold split completed successfully!")
    
    def configure_optuna(self):
        # Define parameters for objective function
        objective_params = {
            'n': self.n,
            'n_y': self.n_y,
            'k': self.k,
            'cv': self.cv,
            'X_trainval': self.X_trainval,
            'y_trainval': self.y_trainval
        }

        load_dotenv()
        db_url = os.getenv('DB_URL')
        
        storage = RDBStorage(
                url=db_url,
                # engine_kwargs={
                #     'pool_pre_ping': True,
                #     'pool_size': 5,
                #     'max_overflow': 10,
                #     'pool_timeout': 10,
                # }
        )

        # Load optuna study and run optimization
        study = optuna.create_study(
                study_name='sleep_stage_classification',
                storage=storage,
                load_if_exists=True,
                sampler=NSGAIISampler(),
                directions=['minimize', 'minimize']
        )

        logging.info("[Main] Optuna study loaded successfully!")

        while True:
            trials = []
            for _ in range(self.batch_size):
                trial = study.ask()
                trials.append(trial)
                logging.info(f"[Trial {trial.number}] New trial received")

            with ProcessPoolExecutor(max_workers=self.batch_size) as executor:
                futures = [executor.submit(objective_func, trial, objective_params) for trial in trials]
                objectives = [future.result() for future in futures]
            logging.info("[Main] All objectives for this batch have been computed")

            for trial, objective in zip(trials, objectives):
                logging.info(f"[Trial {trial.number}] Attempting to save...")
                study.tell(trial, objective)
                logging.info(f"[Trial {trial.number}] Trial successfully saved!")

        '''
        study.optimize(
                lambda trial: objective(trial, objective_params),
                n_trials=self.n_trials,
                n_jobs=self.n_trial_workers,
                callbacks=[MaxTrialsCallback(self.n_trials)],
        )
        '''

    def optimize(self):
        self.load_data()
        self.train_test_split_wrapper()
        self.k_fold()
        self.configure_optuna()

# Define objective and pipeline for optimization
def objective_func(trial, objective_params):
    # Load global parameters
    n = objective_params['n']
    n_y = objective_params['n_y']

    # Feature extraction
    feature_extractor = FeatureExtractor(trial)
    m = feature_extractor.get_m()

    logging.info(f"[Trial {trial.number}] Features extracted")


    # Encode features
    encoder = ColumnTransformer(
        transformers=[
            ('scale', StandardScaler(), ['age']), 
            ('onehot', OneHotEncoder(), ['study', 'sex']),
            ('time', CircularEncoder(), ['lights_off'])
        ]
    )

    logging.info(f"[Trial {trial.number}] Features encoded")

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
    objectives = evaluate_objectives(trial, pipeline, objective_params)
    logging.info(f"[Trial {trial.number}] CV Error: {objectives[0]:.3g}, Memory: {objectives[1]:.0f} MB")
    return objectives

def evaluate_objectives(trial, pipeline, objective_params):
    # Load global parameters
    k = objective_params['k']
    cv = objective_params['cv']
    X_trainval = objective_params['X_trainval']
    y_trainval = objective_params['y_trainval']

    cv_error, cv_memory = np.empty(k), np.empty(k)

    for idx, (train_idx, val_idx) in enumerate(cv.split(X_trainval, y_trainval)):
        logging.info(f"[Trial {trial.number}] Evaluating fold {idx + 1}/{k}...")

        # Split data into training and validation sets
        X_train, y_train = X_trainval[train_idx], y_trainval[train_idx]
        X_val, y_val = X_trainval[val_idx], y_trainval[val_idx]
        
        # Calculate objectives for this fold
        error, memory = evaluate_fold_objectives(pipeline, X_train, y_train, X_val, y_val)
        
        cv_error[idx] = error
        cv_memory[idx] = memory

        logging.info(f"[Trial {trial.number}] Successfully evaluated fold {idx + 1}/{k}!")

    return np.mean(cv_error), np.max(cv_memory)

def evaluate_fold_objectives(pipeline, X_train, y_train, X_val, y_val):
    # Train model
    pipeline.fit(X_train, y_train)

    # Calculate objectives
    error = 1 - pipeline.score(X_val, y_val)
    memory = np.max(memory_usage((pipeline.predict, (X_val,))))
    
    return error, memory
