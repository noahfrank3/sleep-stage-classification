import logging

import numpy as np
from scipy.signal import butter, lfilter
from scipy.signal.windows import hann

# Conditions a signal to be ready for processing
def condition_signal(signal, settings):
    # Unpack settings
    order = settings.order
    jitter = settings.jitter

    # Lowpass filter
    b, a = butter(order, 1 - jitter, btype='low')
    signal = lfilter(b, a, signal)
    logging.debug("Denoising lowpass filter applied to signal")
    
    # Highpass filter
    b, a = butter(order, 1 - jitter, btype='high')
    signal = lfilter(b, a, signal)
    logging.debug("Denoising highpass filter applied to signal")
   
    # Normalize signal
    signal = 2*(signal - np.min(signal))/(np.max(signal) - np.min(signal)) - 1
    logging.debug("Signal normalized")

    # Window signal to remove edge effects
    signal *= hann(len(signal))
    logging.debug("Signal windowed with Hann window")

    return signal

# Processes a full night signal, returns signal snippets as generator
def process_full_signal(signal, annotations, settings):
    for idx, annotation in enumerate(annotations):
        sleep_stage = annotation[-1][-1]

        # Do not include unknown sleep stages and movement time
        if sleep_stage in ['?', 'e']:
            logging.debug(f"Signal with sleep stage '{sleep_stage}' not "
                          f"considered for use")
            continue

        # Sleep stage 4 is classified as sleep stage 3, modern convention
        if sleep_stage == '4':
            logging.debug("Signal with sleep stage '4' reclassified as "
                          "sleep stage '3'")
            sleep_stage = '3'

        # Extract signal snippet
        start_idx = int(annotation[0])
        end_idx = start_idx + int(annotation[1])

        signal_snippet = signal[start_idx:end_idx]
        logging.debug(f"Signal snippet from index {start_idx} to index "
                      f"{end_idx} extracted")

        # Condition signal snippet
        signal_snippet = condition_signal(signal_snippet, settings)
        logging.debug(f"Signal snippet conditioned")

        yield signal_snippet, sleep_stage
