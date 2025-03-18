from pathlib import Path
import pyedflib
import xlrd

import numpy as np
import matplotlib.pyplot as plt

# Basically gonna loop through each PSG, specify metadata true for all excerpts in the PSG, then loop through each annotation within a PSG
# xls file provides metadata, while the edfs provide signal data
# All signals will be appropriately bandpassed
# This is just cassette data

database_path = Path('sleep-edf-database-expanded-1.0.0')

cassette_subjects = xlrd.open_workbook(database_path / 'SC-subjects.xls').sheet_by_index(0)

### Old code, don't remove as I need some of this stuff
'''
sort_edfs = lambda filenames: sorted(filenames, key=lambda s: int(s[3:5]))

## Cassette files
os.chdir(os.path.join('sleep-edf-database-expanded-1.0.0', 'sleep-cassette'))
cassette_files = [f for f in os.listdir()]

# Separate PSGs and hypnograms, nights 1 and 2
cassette_psgs_1 = sort_edfs([s for s in cassette_files if 'PSG' in s and s[5] == '1'])
cassette_psgs_2 = sort_edfs([s for s in cassette_files if 'PSG' in s and s[5] == '2'])
cassette_hypnograms_1 = sort_edfs([s for s in cassette_files if 'Hypnogram' in s and s[5] == '1'])
cassette_hypnograms_2 = sort_edfs([s for s in cassette_files if 'Hypnogram' in s and s[5] == '2'])

print(len(cassette_psgs_1))
print(len(cassette_psgs_2))

file = cassette_hypnograms_1[0]
x = pyedflib.highlevel.read_edf_header(file)
x = x['annotations']

x = pyedflib.EdfReader(file)
x = x.getSignalHeaders()

for dic in x:
    for key, value in dic.items():
        print(f'{key}: {value}')
        print()
'''

### Compute real fourier transform
# from scipy.fft import rfft, rfftfreq
#
# def ft(x, f_s):
#     f = rfftfreq(len(x), 1/f_s)
#     A = 2*np.abs(rfft(x))/len(x)
#
#     return f, A
#
# Sample plot of a signal
# f, A = ft(sig, 100)
# plt.plot(f, A)
# plt.show()
