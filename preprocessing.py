import os
import pyedflib

import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

# Compute real fourier transform
def ft(x, f_s):
    f = rfftfreq(len(x), 1/f_s)
    A = 2*np.abs(rfft(x))/len(x)

    return f, A

sort_edfs = lambda filenames: sorted(filenames, key=lambda s: int(s[3:5]))

# edf extraction example
# file = pyedflib.EdfReader('test.edf')
# sig = file.readSignal(0)

# Sample plot of a signal
# f, A = ft(sig, 100)
# plt.plot(f, A)
# plt.show()

## Cassette files
os.chdir(os.path.join('sleep-edf-database-expanded-1.0.0', 'sleep-cassette'))
cassette_files = [f for f in os.listdir()]

# Separate PSGs and hypnograms, nights 1 and 2
cassette_psgs_1 = sort_edfs([s for s in cassette_files if 'PSG' in s and s[5] == '1'])
cassette_psgs_2 = sort_edfs([s for s in cassette_files if 'PSG' in s and s[5] == '2'])
cassette_hypnograms_1 = sort_edfs([s for s in cassette_files if 'Hypnogram' in s and s[5] == '1'])
cassette_hypnograms_2 = sort_edfs([s for s in cassette_files if 'Hypnogram' in s and s[5] == '2'])

n_birthdate = 0
n_equipment = 0
n_sex = 0
n_technician = 0
for filename in cassette_hypnograms_1 + cassette_hypnograms_2:
    file = pyedflib.EdfReader(filename)

    if file.getBirthdate():
        n_birthdate += 1
    if file.getEquipment():
        n_equipment += 1
    if file.getSex():
        n_sex += 1
    if file.getTechnician():
        n_technician += 1
n = len(cassette_psgs_1 + cassette_psgs_2)
print(f'Birthdate = {100*n_birthdate/n:.3g}%')
print(f'Equipment = {100*n_equipment/n:.3g}%')
print(f'Sex = {100*n_sex/n:.3g}%')
print(f'Technician = {100*n_technician/n:.3g}%')
