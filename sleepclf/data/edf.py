import logging
import os

from pyedflib.highlevel import read_edf

from sleepclf.config import EDF_DATA_PATH

# Get all filenames of a certain type from a directory
def filenames(dir, key):
    filenames = [filename for filename in os.listdir(
        os.path.join(EDF_DATA_PATH, dir)) if key in filename]
    filenames.sort(key=lambda filename: int(filename[3:6]))
    filenames = [os.path.join(
        EDF_DATA_PATH, dir, filename) for filename in filenames]

    logging.debug(f"All filenames from directory '{dir}' of key '{key}' "
                  f"extracted")
    return filenames

# Get all PSG and hypnogram filenames
def all_filenames():
    cas_psgs = filenames('sleep-cassette', 'PSG')
    tel_psgs = filenames('sleep-telemetry', 'PSG')

    cas_hyps = filenames('sleep-cassette', 'Hypnogram')
    tel_hyps = filenames('sleep-telemetry', 'Hypnogram')

    psgs = cas_psgs + tel_psgs
    hyps = cas_hyps + tel_hyps

    logging.debug("All sleep night filenames extracted")
    return (psg for psg in psgs), (hyp for hyp in hyps)

# Extract full signal and annotations for a given night
def extract_full_signal(psg, hyp):
    # Extract signal from PSG
    try:
        signal = read_edf(str(psg))[0][0]
    except OSError:
        logging.warning(f"Failed to extract signal from '{psg}', "
                        f"skipping this signal...")
        return None, None
    
    # Extract annotations from hypnogram
    try:
        annotations = read_edf(str(hyp))[2]['annotations']
    except OSError:
        logging.warning(f"Failed to extract annotations from '{hyp}', "
                        f"skipping this signal...")
        return None, None
    
    logging.debug(f"Signal from '{psg}' and annotations from '{hyp}' "
                  f"extracted")
    return signal, annotations
