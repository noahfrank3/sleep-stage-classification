import numpy as np
from scipy.signal import butter, lfilter
from scipy.signal.windows import tukey

from config.config import CONFIG, new_logger

logger = new_logger('sigproc pre')

# Conditions a signal to be ready for processing
def condition_signal(signal):
    # Unpack settings
    order = CONFIG['sigproc']['filter order']
    jitter = CONFIG['sigproc']['jitter']

    # Denoising filter
    b, a = butter(order, [jitter, 1 - jitter], btype='bandpass')
    signal = lfilter(b, a, signal)
    logger.debug("Denoising filter applied to signal")

    # Zero-padding
    padding = int(0.1*len(signal))
    signal = np.pad(signal, (padding, padding))

    # Windowing
    window = tukey(len(signal), 0.3)
    signal *= window

    # Crop back to original size
    signal = signal[padding:-padding]

    # Normalize signal
    signal /= np.sqrt(np.mean(signal**2))
    logger.debug("Signal normalized")

    return signal

# Processes a full night signal, returns signal snippets as generator
def process_full_signal(signal, annotations):
    for idx, annotation in enumerate(annotations):
        sleep_stage = annotation[-1][-1]

        # Do not include unknown sleep stages and movement time
        if sleep_stage in ['?', 'e']:
            logger.debug(f"Signal with sleep stage '{sleep_stage}' not "
                          f"considered for use")
            continue

        # Sleep stage 4 is classified as sleep stage 3, modern convention
        if sleep_stage == '4':
            logger.debug("Signal with sleep stage '4' reclassified as "
                          "sleep stage '3'")
            sleep_stage = '3'

        # Extract signal snippet
        start_idx = int(annotation[0])
        end_idx = start_idx + int(annotation[1])

        signal_snippet = signal[start_idx:end_idx]
        logger.debug(f"Signal snippet from index {start_idx} to index "
                      f"{end_idx} extracted")

        # Condition signal snippet
        signal_snippet = condition_signal(signal_snippet)
        logger.debug(f"Signal snippet conditioned")

        yield signal_snippet, sleep_stage
