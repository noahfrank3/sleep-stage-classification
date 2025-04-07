import os
from pathlib import Path

import h5py
import numpy as np
from pyedflib.highlevel import read_edf
from scipy.signal import butter, lfilter
from scipy.signal.windows import hamming

def get_signal_filenames(db_path):
    cassette_psg_filenames = get_filenames(db_path, 'sleep-cassette', 'PSG')
    telemetry_psg_filenames = get_filenames(db_path, 'sleep-telemetry', 'PSG')

    cassette_hypnogram_filenames = get_filenames(db_path, 'sleep-cassette',
                                                 'Hypnogram')
    telemetry_hypnogram_filenames = get_filenames(db_path, 'sleep-telemetry',
                                                  'Hypnogram')

    psg_filenames = cassette_psg_filenames + telemetry_psg_filenames
    hypnogram_filenames = cassette_hypnogram_filenames \
            + telemetry_hypnogram_filenames

    return psg_filenames, hypnogram_filenames

def get_filenames(db_path, dir, key):
    filenames = [filename for filename in os.listdir(db_path / dir) \
            if key in filename]
    filenames.sort(key=lambda filename: int(filename[3:6]))
    filenames = [db_path / dir / filename for filename in filenames]

    return filenames

def extract_signals(metadata, psg_filenames, hypnogram_filenames):
    with h5py.File(Path('..') / 'data' / 'data.h5', 'w') as hdf:
        for (idx, metadata_entry), psg_filename, hypnogram_filename in \
                zip(metadata.iterrows(), psg_filenames, hypnogram_filenames):
            # Extract signal and annotations
            try:
                signal = read_edf(str(psg_filename))[0][0]
                annotations = read_edf(str(hypnogram_filename))[2]['annotations']
            except OSError:
                print("Failed to extract signal and annotations")
                continue

            # Store signal and metadata in h5 group
            group = hdf.create_group(str(idx))
            save_signals(group, signal, annotations)
            for key, value in metadata_entry.items():
                group.attrs[key] = value

def save_signals(group, signal, annotations):
    for idx, annotation in enumerate(annotations):
        sleep_stage = annotation[-1][-1]

        # Ignore unknown sleep stages and movement time
        if sleep_stage in ['?', 'e']:
            continue

        # Update sleep stages for modern conventions
        if sleep_stage == '4':
            sleep_stage = '3'

        # Indicies for signal reading
        start_idx = int(annotation[0])
        end_idx = start_idx + int(annotation[1])

        # Condition signal
        signal_snippet = signal[start_idx:end_idx]
        signal_snippet = condition_signal(signal_snippet)

        # Store signal and sleep stage
        dataset = group.create_dataset(str(idx), data=signal_snippet, compression='gzip')
        dataset.attrs['sleep_stage'] = sleep_stage
        
        # Create signal id
        dataset.attrs['id'] = group.name + 'x' + str(idx)

def condition_signal(signal):
    # Lowpass filter
    b, a = butter(5, 0.999, btype='low')
    signal = lfilter(b, a, signal)
    
    # Normalize signal
    signal = 2*(signal - np.min(signal))/(np.max(signal) - np.min(signal)) - 1

    # Window signal to remove edge effects
    signal *= hamming(len(signal))

    return signal

def process_signals(db_path, metadata):
    psg_filenames, hypnogram_filenames = get_signal_filenames(db_path)
    extract_signals(metadata, psg_filenames, hypnogram_filenames)
