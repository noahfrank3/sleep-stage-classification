import h5py
import logging
import os

from .config import DATA_PATH
from .data.edf import all_filenames, extract_full_signal
from .data.hdf5 import save_full_signal
from .data.metadata import create_metadata, load_metadata

HDF_PATH = os.path.join(DATA_PATH, 'data.h5')

def create_hdf5():
    # Load metadata (create if not exists)
    try:
        metadata = load_metadata()
    except FileNotFoundError:
        logging.info("No metadata file found, so creating a new one...")
        metadata = create_metadata()

    # Clear HDF file
    with h5py.File(HDF_PATH, 'w') as hdf:
        logging.info("HDF data file cleared")

    # Save each signal to the hdf5
    psgs, hyps = all_filenames()
    for psg, hyp, (idx, entry) in zip(psgs, hyps, metadata.iterrows()):
        signal, annotations = extract_full_signal(psg, hyp)

        if signal is not None and annotations is not None:
            save_full_signal(idx, signal, annotations, entry)

if __name__ == '__main__':
    logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s - %(message)s'
    )
    create_hdf5()
