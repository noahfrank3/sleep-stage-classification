from optimization import Optimizer

if __name__ == '__main__':
    hpc_enabled = True
    k = 5
    n_trials = 10000
    f_trial_workers = 0.8

    optimizer = Optimizer(hpc_enabled, k, n_trials, f_trial_workers)
    optimizer.optimize()
