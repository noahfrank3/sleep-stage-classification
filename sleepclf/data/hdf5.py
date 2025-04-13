import logging
import os

import h5py

from sleepclf.config import DATA_PATH, SIGPROC_SETTINGS
from sleepclf.sigproc.preprocess import process_full_signal

HDF_PATH = os.path.join(DATA_PATH, 'data.h5')

def create_group(hdf, idx, metadata_entry):
    group = hdf.create_group('g' + str(idx))

    for key, value in metadata_entry.items():
        group.attrs[key] = value

    logging.debug(f"Group {group.name} created")
    return group

def create_dataset(hdf, group, idx, signal_snippet, sleep_stage):
    dataset = group.create_dataset(
            'd' + str(idx),
            data=signal_snippet,
            compression='gzip'
    )

    dataset.attrs['sleep_stage'] = sleep_stage
    dataset.attrs['id'] = group.name + 'x' + str(idx)

    logging.debug(f"Dataset {dataset.name} with signal ID "
                  f"{dataset.attrs['id']} created for group {group.name}")
    return dataset

def save_full_signal(idx, signal, annotations, metadata_entry):
    with h5py.File(HDF_PATH, 'a') as hdf:
        signal_snippets = process_full_signal(
                signal,
                annotations, 
                SIGPROC_SETTINGS
        )
        group = create_group(hdf, idx, metadata_entry)
        for idx, (signal_snippet, sleep_stage) in enumerate(signal_snippets):
            create_dataset(hdf, group, idx, signal_snippet, sleep_stage)
