import os

from dotenv import load_dotenv
import optuna
from optuna.storages import RDBStorage
from optuna.visualization import plot_pareto_front

if __name__ == '__main__':
    load_dotenv()
    db_url = os.getenv('DB_URL')

    storage = RDBStorage(url=db_url)
    study = optuna.load_study(study_name='sleep_stage_classification', storage=storage)

    df = study.trials_dataframe()
    print(df)

    fig = plot_pareto_front(study, target_names=['CV Error', 'Memory (MB)'])
    fig.show()
