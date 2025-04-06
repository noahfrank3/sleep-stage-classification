import pandas as pd
from pathlib import Path
from metadata import (
        CassettePreprocessor,
        TelemetryPreprocessor,
        circularize_time
)
from signal_data import get_signal_filenames, extract_signals

db_path_mappings = {
    'zip': Path('..', 'sleep-edf-database-expanded-1.0.0'),
    'wget': Path('..', 'physionet.org', 'files', 'sleep-edfx', '1.0.0')
}
study_name = None

if __name__ == '__main__':
    # Ask for download type
    while True:
        download_type = input("What method did you use to download the "
                              "database? zip or wget? ")

        if download_type in ['zip', 'wget']:
            break
        else:
            print("You must specify either 'zip' or 'wget'")

    # Create database path
    db_path = db_path_mappings[download_type]

    # Create metadata preprocessors
    cassette_preprocessor = CassettePreprocessor(db_path)
    telemtry_preprocessor = TelemetryPreprocessor(db_path)

    # Preprocess cassette and telemtry metadata
    cassette_data = cassette_preprocessor.get_data()
    telemetry_data = telemtry_preprocessor.get_data()

    # Combine cassette and telemtry metadata
    metadata = pd.concat([cassette_data, telemetry_data], ignore_index=True)

    # Circularize time
    metadata = circularize_time(metadata)

    # Save data to csv
    # metadata.to_csv(Path('..') / 'data' / 'metadata.csv', index=False)

    # Get filenames for signal extraction
    psg_filenames, hypnogram_filenames = get_signal_filenames(db_path)
    
    # Extract signals
    extract_signals(metadata, psg_filenames, hypnogram_filenames)
