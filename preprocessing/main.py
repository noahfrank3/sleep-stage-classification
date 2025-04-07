from pathlib import Path

from metadata import get_metadata
from signal_data import process_signals

def get_db_path():
    db_path_mappings = {
        'zip': Path('..', 'sleep-edf-database-expanded-1.0.0'),
        'wget': Path('..', 'physionet.org', 'files', 'sleep-edfx', '1.0.0')
    }

    # Ask for download type
    while True:
        download_type = input("What method did you use to download the "
                              "database? zip or wget? ")

        if download_type in ['zip', 'wget']:
            break
        else:
            print("You must specify either 'zip' or 'wget'")

    # Create database path
    return db_path_mappings[download_type]

if __name__ == '__main__':
    db_path = get_db_path()
    metadata = get_metadata(db_path)
    process_signals(db_path, metadata)
