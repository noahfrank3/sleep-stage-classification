import os
import pyedflib

import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

# edf extraction example
# file = pyedflib.EdfReader('test.edf')
# sig = file.readSignal(0)

# Compute real fourier transform
def ft(x, f_s):
    f = rfftfreq(len(x), 1/f_s)
    A = 2*np.abs(rfft(x))/len(x)

    return f, A

# Sample plot of a signal
# f, A = ft(sig, 100)
# plt.plot(f, A)
# plt.show()

sort_edfs = lambda filenames: sorted(filenames, key=lambda s: int(s[3:5]))

## Cassette files
os.chdir(os.path.join('sleep-edf-database-expanded-1.0.0', 'sleep-cassette'))
cassette_files = [f for f in os.listdir()]

# Separate PSGs and hypnograms, nights 1 and 2
cassette_psgs_1 = sort_edfs([s for s in cassette_files if 'PSG' in s and s[5] == '1'])
cassette_psgs_2 = sort_edfs([s for s in cassette_files if 'PSG' in s and s[5] == '2'])
cassette_hypnograms_1 = sort_edfs([s for s in cassette_files if 'Hypnogram' in s and s[5] == '1'])
cassette_hypnograms_2 = sort_edfs([s for s in cassette_files if 'Hypnogram' in s and s[5] == '2'])
