import os
import re
import pyedflib
import pandas as pd
import threading
import logging
from collections import deque
from pathlib import Path
from utils import *

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Set path to database
database_path = Path('..', 'physionet.org', 'files', 'sleep-edfx', '1.0.0')
# database_path = Path('..', 'sleep-edf-database-expanded-1.0.0')

def preprocess_cassette_subject_data():
    # Read data from xls file
    subject_data = pd.read_excel(database_path / 'SC-subjects.xls')

    # Rename columns
    subject_data = subject_data.rename(columns={
        'k': 'subject',
        'sex (F=1)': 'sex',
        'LightsOff': 'lights_off'
    })

    # Define study
    subject_data['study'] = 'cassette'
    
    # Fix sex labels
    subject_data['sex'] = subject_data['sex'].astype(str)

    subject_data.loc[subject_data['sex'] == '1', 'sex'] = 'F'
    subject_data.loc[subject_data['sex'] == '2', 'sex'] = 'M'

    return subject_data

def preprocess_telemetry_subject_data():
    # Read data from xls file
    subject_data = pd.read_excel(database_path / 'ST-subjects.xls', skiprows=1)
    
    # Rename columns
    subject_data = subject_data.rename(columns={
        'Nr': 'subject',
        'Age': 'age',
        'M1/F2': 'sex',
        'night nr': 'placebo_night',
        'lights off': 'placebo_lights_off',
        'night nr.1': 'temazepam_night',
        'lights off.1': 'temazepam_lights_off'
    })

    # Split data into placebo and temazepam data
    placebo_data = subject_data.drop(['temazepam_night', 'temazepam_lights_off'],
                             axis=1)
    temazepam_data = subject_data.drop(['placebo_night', 'placebo_lights_off'],
                               axis=1)

    placebo_data = placebo_data.rename(columns={
        'placebo_night': 'night',
        'placebo_lights_off': 'lights_off'
    })
    temazepam_data = temazepam_data.rename(columns={
        'temazepam_night': 'night',
        'temazepam_lights_off': 'lights_off'
    })

    placebo_data['study'] = 'placebo'
    temazepam_data['study'] = 'temazepam'
    
    # Recombine placebo and temazepam data
    subject_data = pd.concat([placebo_data, temazepam_data],
                             ignore_index=True)
    subject_data = subject_data.sort_values(by=['subject', 'night'],
                                            ignore_index=True)
    
    # Fix sex labels
    subject_data['sex'] = subject_data['sex'].astype(str)

    subject_data.loc[subject_data['sex'] == '1', 'sex'] = 'M'
    subject_data.loc[subject_data['sex'] == '2', 'sex'] = 'F'

    return subject_data

def preprocess_subject_data(subject_data, study_name):
    data_path = database_path / ('sleep-' + study_name)

    # Reorder columns
    subject_data = subject_data[[
        'study',
        'subject',
        'night',
        'age',
        'sex',
        'lights_off'
    ]]

    # Circularize time
    time_encoder = CircularEncoder()
    
    subject_data['lights_off'] = subject_data['lights_off'].apply(
            lambda t: (t.hour + t.minute/60)/24)
    subject_data['lights_off_cos'], subject_data['lights_off_sin'] = \
                time_encoder.transform(subject_data['lights_off'])
    del subject_data['lights_off']

    # Get PSG filenames
    psg_filenames = [s for s in os.listdir(
        data_path) if 'PSG' in s]
    psg_filenames.sort(key=lambda s: int(s[3:6]))
    subject_data['psg_filename'] = psg_filenames

    # Get hypnogram filenames
    hypnogram_filenames = [s for s in os.listdir(
        data_path) if 'Hypnogram' in s]
    hypnogram_filenames.sort(key=lambda s: int(s[3:6]))
    subject_data['hypnogram_filename'] = hypnogram_filenames
    
    # Get technicians
    subject_data['technician'] = [s[7] for s in subject_data[
        'hypnogram_filename']]

    del subject_data['subject']
    return subject_data

# Get signal data
def get_signal_data(subject_data, study_name):
    data_path = database_path / ('sleep-' + study_name)
    min_freq = 0
    max_freq = SAMPLE_FREQ/2

    # Log status
    status_deque = deque(maxlen=1)
    end_var = 0
    log_thread = threading.Thread(
            target=log_status,
            args=(status_deque,),
            kwargs={'end_var': end_var}
    )
    log_thread.start()

    # Loop through each subject
    data = []
    n_subjects = len(subject_data)

    for subject_idx, subject_row in subject_data.iterrows():
        # Unpack filenames
        psg_filename = subject_row['psg_filename']
        hypnogram_filename = subject_row['hypnogram_filename']

        # Extract annotations
        try:
            annotations = pyedflib.highlevel.read_edf_header(
                str(data_path / hypnogram_filename))['annotations']
        except OSError:
            logging.info(f'Failed to extract {study_name} subject '
                         f'{subject_idx}')
            continue

        # Convert subject row to list
        subject_row = subject_row.tolist()

        # Get LP and HP prefilter frequencies
        prefilter_text = pyedflib.highlevel.read_edf(
                str(data_path / psg_filename))[1][0]['prefilter']
        
        LP_freq = re.search(r'LP:([\d.]+)', prefilter_text)
        if LP_freq is not None:
            LP_freq = float(LP_freq.group(1))
            min_freq = max(min_freq, LP_freq)
        
        HP_freq = re.search(r'HP:([\d.]+)', prefilter_text)
        if HP_freq is not None:
            HP_freq = float(HP_freq.group(1))
            max_freq = min(max_freq, HP_freq)
        
        # Add data for each annotation
        n_stages = len(annotations)

        for stage_idx, annotation in enumerate(annotations):
            # Print status
            status = f'Processing {study_name} night ' + \
                     f'{subject_idx}/{n_subjects}, ' + \
                     f'stage {stage_idx}/{n_stages}'
            status_deque.append(status)

            # Sleep stage label
            sleep_stage = annotation[-1][-1]
            
            # Ignore unknown stages and movement time
            if sleep_stage in ['W', 'e']:
                continue

            # Same sleep stage
            if sleep_stage == '4':
                sleep_stage = '3'

            # Indices for signal reading
            start_index = int(annotation[0])
            duration = int(annotation[1])

            end_index = start_index + duration
            
            # Add sleep stage to data
            data_row = subject_row + [start_index, end_index, sleep_stage]
            data.append(data_row)

            # Check for short-duration sleep stages
            min_freq = max(min_freq, SAMPLE_FREQ/duration)

    # End status printing thread
    status_deque.append(end_var)
    log_thread.join()

    # Create dataframe
    column_names = subject_data.columns.tolist()
    column_names += ['start_index', 'end_index', 'sleep_stage']

    data = pd.DataFrame(data, columns=column_names)
    return data

def preprocessing_pipeline(study_name):
    # Get subject data
    logging.info(f'Preprocessing {study_name} subject data...')

    match study_name:
        case 'cassette':
            subject_data = preprocess_cassette_subject_data()
        case 'telemetry':
            subject_data = preprocess_telemetry_subject_data()

    subject_data = preprocess_subject_data(subject_data, study_name)
    logging.info(f'{study_name.capitalize()} '
                 f'subject data processed successfully')

    # Get signal data
    data = get_signal_data(subject_data, study_name)
    return data

# Create data
if __name__ == '__main__':
    # Initialize min/max frequencies
    min_freq = 0
    max_freq = SAMPLE_FREQ/2

    # Get cassette and telemtry data
    cassette_data = preprocessing_pipeline('cassette')
    telemetry_data = preprocessing_pipeline('telemetry')

    print(cassette_data)
    print(telemetry_data)

    # Combine cassette and telemtry data
    data = pd.concat([cassette_data, telemetry_data], ignore_index=True)

    X = data
    y = X.pop('sleep_stage')
    technicians = X.pop('technician')
    del X['hypnogram_filename']

    # Save data
    path = Path('../data')
    X.to_csv(path / 'X.csv', index=False)
    y.to_csv(path / 'y.csv', index=False)
    technicians.to_csv(path / 'technicians.csv', index=False)
