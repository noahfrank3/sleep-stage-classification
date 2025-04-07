import os

from dotenv import load_dotenv
import optuna
from optuna.samplers import NSGAIISampler

if __name__ == '__main__':
    load_dotenv()
    db_url = os.getenv('DB_URL')

    optuna.delete_study(
        study_name='sleep_stage_classification', 
        storage=db_url
    )

    study = optuna.create_study(
            study_name='sleep_stage_classification',
            storage=db_url,
            load_if_exists=True,
            sampler=NSGAIISampler(),
            directions=['minimize', 'minimize']
    )
