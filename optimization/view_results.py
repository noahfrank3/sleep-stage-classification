import os

from dotenv import load_dotenv
import optuna
from optuna.visualization import plot_pareto_front

load_dotenv()
db_url = os.getenv('DB_URL')

study = optuna.load_study(study_name='sleep_stage_classification', storage=db_url)
fig = plot_pareto_front(study)
fig.show()
