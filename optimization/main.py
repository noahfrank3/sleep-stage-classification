import logging
from multiprocessing import cpu_count
import os
from pathlib import Path

from dotenv import load_dotenv

from optimization import run_optimization
from reset_db import reset_db

# Parameters
hpc_enabled = True # enable HPC
test_db = False # use test database
do_reset_db = False # reset database
reset_only = False # only reset database

k = 5 # number of folds for k-fold CV
n_trials = 10000 # number of trials for optimization run
f_trial_workers = 0.65 # fraction of workers dedicated to running whole trials (HPC only)

data_path = Path('..') / 'data' # path of data directory

if hpc_enabled:
    n_workers = cpu_count() # number of CPUs to use
    n_trial_workers = int(f_trial_workers*n_workers)
    n_internal_workers = n_workers - n_trial_workers
else:
    n_trial_workers = 1
    n_internal_workers = 1

if test_db:
    db_url = 'sqlite:///test.db'
else:
    load_dotenv()
    db_url = os.getenv('DB_URL')

if (do_reset_db or reset_only) and not hpc_enabled:
    reset_db(db_url)

# Save global parameters
global_params = {
        'k': k,
        'n_trials': n_trials,
        'n_trial_workers': n_trial_workers,
        'n_internal_workers': n_internal_workers,
        'data_path': data_path,
        'db_url': db_url
}

# Configure logger
if hpc_enabled:
    logging.basicConfig()
    logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Run optimization
if not reset_only:
    run_optimization(global_params)
