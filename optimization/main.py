from optimization import Optimizer

if __name__ == '__main__':
    hpc_enabled = True
    n_cpus = 24
    k = 5
    n_trials = 10000
    f_trial_workers = 0.8

    optimizer = Optimizer(hpc_enabled, n_cpus, k, n_trials, f_trial_workers)
    optimizer.optimize()
