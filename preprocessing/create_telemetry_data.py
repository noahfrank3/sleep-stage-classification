import os
import sys
import re
import threading
import pyedflib
import pandas as pd

sys.path.append('..')

from pathlib import Path
from collections import deque
from utils import CircularEncoder, print_status

### NOT READY FOR USE
def create_telemetry_data():
    # Create path labels
    database_path = Path('..', 'sleep-edf-database-expanded-1.0.0')
    telemetry_path = database_path / 'sleep-telemetry'
    
    # Sampling frequency
    sample_freq = 100 # Hz

    global_min_freq = 0
    global_max_freq = sample_freq/2

    # Get subject data
    telemetry_data = pd.read_excel(database_path / 'ST-subjects.xls')
    telemetry_data = telemetry_data.rename(columns={'k': 'subject', 'sex (F=1)': 'sex', 'LightsOff': 'lights_off'})
    
    # Make sex labels readable
    telemetry_data['sex'] = telemetry_data['sex'].astype(str)

    telemetry_data.loc[telemetry_data['sex'] == '0', 'sex'] = 'M'
    telemetry_data.loc[telemetry_data['sex'] == '1', 'sex'] = 'F'

    # Circularize time
    time_encoder = CircularEncoder()

    telemetry_data['lights_off'] = telemetry_data['lights_off'].apply(lambda t: (t.hour + t.minute/60)/24)
    telemetry_data['lights_off_cos'], telemetry_data['lights_off_sin'] = time_encoder.transform(telemetry_data['lights_off'])
    del telemetry_data['lights_off']

    # Get PSG filenames
    telemetry_psgs = [s for s in os.listdir(telemetry_path) if 'PSG' in s]
    telemetry_psgs.sort(key=lambda s: int(s[3:6]))
    telemetry_data['psg_filename'] = telemetry_psgs

    # Get hypnogram filenames
    telemetry_hypnograms = [s for s in os.listdir(telemetry_path) if 'Hypnogram' in s]
    telemetry_hypnograms.sort(key=lambda s: int(s[3:6]))
    telemetry_data['hypnogram_filename'] = telemetry_hypnograms

    # Get technicians
    telemetry_data['technician'] = [s[7] for s in telemetry_data['hypnogram_filename']]

    # Begin status printing thread
    status_deque = deque(maxlen=1)
    end_var = 0

    status_thread = threading.Thread(target=print_status, args=(status_deque, 0))
    status_thread.start()

    # Add EEG signal data
    telemetry_data_temp = []
    n_nights = len(telemetry_data)

    for night_idx, base_row in telemetry_data.iterrows():
        # Unpack and repack features
        psg_filename = base_row['psg_filename']
        hypnogram_filename = base_row['hypnogram_filename']
        base_row = base_row.tolist()
        
        # Adjust min and max frequencies with LP and HP prefilter frequencies
        min_freq = global_min_freq
        max_freq = global_max_freq

        prefilter_text = pyedflib.highlevel.read_edf(str(telemetry_path / psg_filename))[1][0]['prefilter']
        
        LP_freq = re.search(r'LP:([\d.]+)', prefilter_text)
        if LP_freq != None:
            LP_freq = float(LP_freq.group(1))
            max_freq = LP_freq if LP_freq < global_max_freq else global_max_freq
        
        HP_freq = re.search(r'HP:([\d.]+)', prefilter_text)
        if HP_freq != None:
            HP_freq = float(HP_freq.group(1))
            min_freq = HP_freq if HP_freq > global_min_freq else global_min_freq
        
        # Add data for each annotation
        annotations = pyedflib.highlevel.read_edf_header(str(telemetry_path / hypnogram_filename))['annotations']
        n_stages = len(annotations) - 1

        for stage_idx, annotation in enumerate(annotations[:-1]):
            # Print status
            status = f'[preprocessing.py]: Processing night {night_idx}/{n_nights}, stage {stage_idx}/{n_stages}'
            status_deque.append(status)

            # Indices for signal reading
            start_index = int(annotation[0])
            end_index = start_index + int(annotation[1])
            
            # Check for short-duration sleep stages
            duration_freq = sample_freq/int(annotation[1])
            min_freq = max(min_freq, duration_freq)
            
            # Sleep stage label
            label = annotation[-1][-1]

            # Add this data point
            new_row = base_row[:]
            new_row.extend((start_index, end_index, min_freq, max_freq, label))
            telemetry_data_temp.append(new_row)

    # End status printing thread
    status_deque.append(end_var)
    status_thread.join()

    # Reconstruct dataframe
    column_names = telemetry_data.columns.tolist()
    column_names.extend(('start_index', 'end_index', 'min_freq', 'max_freq', 'label'))

    telemetry_data = pd.DataFrame(telemetry_data_temp, columns=column_names)

    # Add indicator for cassette/telemetry study
    telemetry_data['study'] = 1

    return telemetry_data
