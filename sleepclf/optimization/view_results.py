import os

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.storages import RDBStorage
from optuna.visualization import plot_hypervolume_history, plot_param_importances, plot_parallel_coordinate
import pandas as pd
from plotly.io import show

clf_mappings = {
    'QDA': 'deepskyblue',
    'LR': 'darkorange',
    'NB': 'forestgreen',
    'DT': 'crimson',
    'RF': 'goldenrod',
    'kNN': 'mediumslateblue',
    'SVM': 'darkmagenta',
    'XGB': 'dodgerblue',
    'NN': 'slategray'
}

dim_reduction_mappings = {
        'None': 'deepskyblue',
        'KPCA': 'darkorange',
        'LDA': 'forestgreen',
        'SVD': 'crimson',
        'LASSO': 'goldenrod'
}

def get_study():
    load_dotenv()
    db_url = os.getenv('DB_URL')

    return optuna.load_study(
            study_name='sleep_stage_classification',
            storage=RDBStorage(url=db_url)
    )

def get_all_data(study):
    data = study.trials_dataframe(attrs=('state', 'params', 'value'))
    data = data[data['state'] == 'COMPLETE']
    data = data.rename(columns={
        'params_clf': 'clf',
        'params_dim_reduction': 'dim_reduction',
        'values_0': 'cv_error',
        'values_1': 'memory'
    })
    
    data_1 = data[data['memory'] < 2000]
    data_2 = data[data['memory'] >= 2000]

    data_2['memory'] -= data_2['memory'].mean()
    data_2['memory'] /= np.std(data_2['memory'], ddof=1)
    data_2['memory'] *= np.std(data_1['memory'], ddof=1)
    data_2['memory'] += data_1['memory'].mean()

    data = pd.concat([data_1, data_2], ignore_index=True)

    return data[['clf', 'dim_reduction', 'cv_error', 'memory']]

def get_pareto_data(study):
    data = []
    for trial in study.best_trials:
        data.append({
            'clf': trial.params['clf'],
            'dim_reduction': trial.params['dim_reduction'],
            'cv_error': trial.values[0],
            'memory': trial.values[1]
        })
    return pd.DataFrame(data)

def get_data_by_clf(data):
    old_data = data
    data = {}
    for clf in clf_mappings.keys():
        data[clf] = old_data[old_data['clf'] == clf]
    return data

def plot_data(data, title):
    data = get_data_by_clf(data)

    fig, ax = plt.subplots()
    for clf, subdata in data.items():
        ax.scatter(subdata['cv_error'], subdata['memory'], color=clf_mappings[clf], label=clf, s=15)
    ax.set_title(title, fontsize='large')
    ax.set_xlabel("CV Error", fontsize='large')
    ax.set_ylabel("Memory (MB)", fontsize='large')
    ax.legend(fontsize='large')
    ax.grid(True)
    fig.savefig(f'{title.lower().replace(' ', '_')}.svg')

if __name__ == '__main__':
    study = get_study()

    all_data = get_all_data(study)
    plot_data(all_data, 'All Data')

    pareto_data = get_pareto_data(study)
    plot_data(pareto_data, 'Pareto Frontier')

    print(f"Number of completed trials: {len(all_data)}")

    # fig = plot_parallel_coordinate(study, target=lambda t: t.values[0], target_name='CV Error', params=['clf', 'dim_reduction'])
    # fig = plot_param_importances(study)
    fig = plot_hypervolume_history(study, (1, 1024*4))
    show(fig)
