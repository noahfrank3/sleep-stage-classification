import logging

import numpy as np
from scipy.signal import butter, lfilter, welch

# Calculate power of a signal over a certain frequency range
def power(signal, f_low, f_high, settings):
    # Unpack settings
    fs = settings.fs
    order = settings.order
    jitter = settings.jitter

    # Calculate Nyquist frequency
    f_n = fs / 2
    
    # Normalize frequencies and add jitter
    f_low /= f_n
    f_high /= f_n

    f_low += jitter
    f_high -= jitter

    # Filter signal
    b, a = butter(order, (f_low, f_high), btype='band')
    signal = lfilter(b, a, signal)
    logging.debug("Bandpass filter applied to signal")
    
    # Calculate power
    _, power = welch(signal, fs)
    power = np.mean(power)

    logging.debug(f"Signal power is {power:.3g}")
    return power

# Define signal ranges to filter over
def ranges(n_divs_map, f_sleep_map):
    ranges = []
    for stage, n_divs in n_divs_map.items():
        f_min, f_max = f_sleep_map[stage]
        f_range = (f_max - f_min)/n_divs

        for div in range(n_divs):
            f_low = f_min + div*f_range
            f_high = f_low + f_range

            ranges.append((f'{stage}_{div}', (f_low, f_high)))
    return ranges

# Get all RMS values for signal
def get_amps(signal, ranges):
    signal_data = {}
    for header, (f_low, f_high) in ranges:
        amp = get_amp(signal, f_low, f_high)
        signal_data[header] = amp
    return signal_data

# Get signal features
def get_signal_features(signal, n_divs_map):
    ranges = get_ranges(n_divs_map)
    return get_amps(signal, ranges)
