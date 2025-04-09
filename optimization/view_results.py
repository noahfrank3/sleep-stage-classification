from pathlib import Path

import optuna
from optuna.visualization import plot_pareto_front

if __name__ == '__main__':
    storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(
                str(Path('..') / 'data' / 'optuna_data.log')))

    study = optuna.load_study(study_name='sleep_stage_classification', storage=storage)

    df = study.trials_dataframe()
    print(df)

    fig = plot_pareto_front(study, target_names=['CV Error', 'Memory (MB)'])
    fig.show()
