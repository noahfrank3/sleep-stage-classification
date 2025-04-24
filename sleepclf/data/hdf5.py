import h5py

from config.config import CONFIG, new_logger
from sleepclf.sigproc.preprocess import process_full_signal

HDF_PATH = CONFIG['paths']['data'] / 'data.h5'
logger = new_logger('hdf5')

def create_group(hdf, idx, metadata_entry):
    group = hdf.create_group('g' + str(idx))

    for key, value in metadata_entry.items():
        group.attrs[key] = value

    logger.debug(f"Group {group.name} created")
    return group

def create_dataset(hdf, group, idx, signal_snippet, sleep_stage):
    dataset = group.create_dataset(
            'd' + str(idx),
            data=signal_snippet,
            compression='gzip'
    )

    dataset.attrs['sleep_stage'] = sleep_stage
    dataset.attrs['id'] = group.name + 'x' + str(idx)

    logger.debug(f"Dataset {dataset.name} with signal ID "
                  f"{dataset.attrs['id']} created for group {group.name}")
    return dataset

def save_full_signal(idx, signal, annotations, metadata_entry):
    with h5py.File(HDF_PATH, 'a') as hdf:
        signal_snippets = process_full_signal(signal, annotations)
        group = create_group(hdf, idx, metadata_entry)
        for idx, (signal_snippet, sleep_stage) in enumerate(signal_snippets):
            create_dataset(hdf, group, idx, signal_snippet, sleep_stage)
