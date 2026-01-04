import json
import logging
from pathlib import Path
import subprocess

paths = {}

def get_root_path():
    return Path(
            subprocess.check_output(
                ['git', 'rev-parse', '--show-toplevel'],
                universal_newlines=True
            )
            .strip()
    )
paths['root'] = get_root_path()

config_path = paths['root'] / 'config' / 'config.json'
def get_config():
    with open(config_path, 'r') as file:
        return json.load(file)
config = get_config()

def get_original_db_path():
    for path in config['original database paths']:
        path = Path(path)
        if path.is_dir():
            return path
paths['original_db'] = get_original_db_path()

paths['data'] = paths['root'] / 'data'
paths['output'] = paths['root'] / 'output'

config['paths'] = paths
CONFIG = config

class NameFilter(logging.Filter):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def filter(self, record):
        record.modulename = self.name
        return True

LOGGING_LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
}

default_logging_level = LOGGING_LEVELS[CONFIG['logging']['default level']]

def new_logger(name, level=default_logging_level):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(modulename)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.addFilter(NameFilter(name))

    logger.addHandler(handler)
    return logger
