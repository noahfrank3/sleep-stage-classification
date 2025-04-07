import numpy as np
from scipy.signal import butter, lfilter

fs = 100
f_sleep_map = {
            'theta': (4, 7.5),
            'alpha': (8, 13),
            'beta': (14, 30),
            'gamma': (30, 50)
}

# Define signal ranges to filter over
def get_ranges(n_divs_map):
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

# Get RMS value for signal in certain range
def get_amp(signal, f_low, f_high):
    f_n = fs / 2
    
    f_low /= f_n
    f_high /= f_n

    f_low += 0.001
    f_high -= 0.001

    b, a = butter(5, (f_low, f_high), btype='band')
    signal = lfilter(b, a, signal)
    return np.sqrt(np.mean(signal**2))

# Get signal features
def get_signal_features(signal, n_divs_map):
    ranges = get_ranges(n_divs_map)
    return get_amps(signal, ranges)
