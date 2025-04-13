import os

from .sigproc.settings import SigprocSettings

### Raw configuration

# Data files
DOWNLOAD_METHOD = 'wget' # method of downloading dataset, 'wget' or 'zip'

# Signal processing
FS = 100 # sampling frequency, Hz
JITTER = 0.001 # jitter for managing singularities
FILTER_ORDER = 5 # order of all butterworth filters

# Maps sleep wave type to its frequency range
F_SLEEP_MAP = {
            'theta': (4, 7.5),
            'alpha': (8, 13),
            'beta': (14, 30),
            'gamma': (30, 50)
}


### Derived configuration

# Data files
DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '../data'
))
OUTPUT_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    '../output'
))

download_method_map = {
        'wget': 'physionet.org/files/sleep-edfx/1.0.0',
        'zip': 'sleep-edf-database-expanded-1.0.0'
}
EDF_DATA_PATH = os.path.abspath(os.path.join(
    DATA_PATH,
    download_method_map[DOWNLOAD_METHOD]
))

# Signal processing
SIGPROC_SETTINGS = SigprocSettings(FS, FILTER_ORDER, JITTER)
