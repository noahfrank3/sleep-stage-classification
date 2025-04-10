import os

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.storages import RDBStorage
from optuna.visualization import plot_pareto_front

if __name__ == '__main__':
    load_dotenv()
    db_url = os.getenv('DB_URL')

    storage = RDBStorage(url=db_url)
    study = optuna.load_study(study_name='sleep_stage_classification', storage=storage)

    df = study.trials_dataframe(attrs=('state', 'value'))
    df = df.rename(columns={
        'state': 'Status',
        'value_0': 'CV Error',
        'value_1': 'Memory (MB)'
    })

    print(df)

    cv_errors = []
    memorys = []
    for trial in study.best_trials:
        cv_errors.append(trial.values[0])
        memorys.append(trial.values[1])
    cv_errors = np.array(cv_errors)
    memorys = np.array(memorys)

    print(cv_errors)
    print(memorys)

    fig, ax = plt.subplots()
    ax.scatter(cv_errors, memorys, color='dodgerblue')
    ax.set_title("Pareto Frontier", fontsize='large')
    ax.set_xlabel("CV Error", fontsize='large')
    ax.set_ylabel("Memory (MB)", fontsize='large')
    ax.grid(True)
    fig.show()
