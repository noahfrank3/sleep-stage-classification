import h5py
from joblib import Parallel, delayed
import logging
import os

from .config import HPC_PREPROCESSING, HPC_PREPROCESSING_JOBS, DATA_PATH
from .data.edf import all_filenames, extract_full_signal
from .data.hdf5 import save_full_signal
from .data.metadata import create_metadata, load_metadata

HDF_PATH = os.path.join(DATA_PATH, 'data.h5')

def extract_and_save_full_signal(psg, hyp, idx, entry):
    signal, annotations = extract_full_signal(psg, hyp)

    if signal is not None and annotations is not None:
        save_full_signal(idx, signal, annotations, entry)

def create_hdf5():
    # Clear HDF file
    with h5py.File(HDF_PATH, 'w') as hdf:
        logging.info("HDF data file cleared")

    # Load metadata (create if not exists)
    try:
        metadata = load_metadata()
    except FileNotFoundError:
        logging.info("No metadata file found, so creating a new one...")
        metadata = create_metadata()

    # Extract PSG and hypnogram filenames
    psgs, hyps = all_filenames()

    # Save each signal to the hdf5
    if HPC_PREPROCESSING:
        Parallel(n_jobs=HPC_PREPROCESSING_JOBS) \
                (delayed(extract_and_save_full_signal) \
                (psg, hyp, idx, entry) for psg, hyp, (idx, entry) \
                in zip(psgs, hyps, metadata.iterrows()))
    else:
        for psg, hyp, (idx, entry) in zip(psgs, hyps, metadata.iterrows()):
            extract_and_save_full_signal(psg, hyp, idx, entry)

if __name__ == '__main__':
    logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s - %(message)s'
    )
    create_hdf5()
