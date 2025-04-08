from pathlib import Path
from multiprocessing import cpu_count
import os

from dotenv import load_dotenv

from optimization import run_optimization

if __name__ == '__main__':
    # Parameters
    k = 5 # number of folds for k-fold CV
    n_trials = 10000 # number of trials for optimization run
    n_workers = cpu_count() # number of CPUs to use (set to cpu_count() for HPC)
    f_trial_workers = 0.65 # fraction of workers dedicated to running whole trials
    data_path = Path('..') / 'data' # path of data directory

    n_trial_workers = int(f_trial_workers*n_workers)
    n_trial_workers = 1
    n_internal_workers = n_workers - n_trial_workers

    load_dotenv()
    db_url = os.getenv('DB_URL')

    # Run optimization
    run_optimization(data_path, k, n_trials, n_trial_workers, n_internal_workers, db_url)
