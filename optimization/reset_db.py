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

    study = optuna.create_study(
            study_name='sleep_stage_classification',
            storage=db_url,
            sampler=NSGAIISampler(),
            directions=['minimize', 'minimize']
    )
