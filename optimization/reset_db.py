import os

from dotenv import load_dotenv
import optuna
from optuna.samplers import NSGAIISampler

def reset_db(db_url):
    try:
        optuna.delete_study(
            study_name='sleep_stage_classification', 
            storage=db_url
        )
    except:
        pass

    optuna.create_study(
            study_name='sleep_stage_classification',
            storage=db_url,
            sampler=NSGAIISampler(population_size=200),
            directions=['minimize', 'minimize']
    )

if __name__ == '__main__':
    load_dotenv()
    db_url = os.getenv('DB_URL')

    reset_db(db_url)
