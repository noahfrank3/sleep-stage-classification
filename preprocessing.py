import os
import pyedflib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from circular_encoder import CircularEncoder

### Cassette Data
cassette_path = os.path.join('sleep-edf-database-expanded-1.0.0', 'sleep-cassette')

# Get subject data
cassette_subject_data_path = os.path.join('sleep-edf-database-expanded-1.0.0', 'SC-subjects.xls')
cassette_data = pd.read_excel(cassette_subject_data_path)

cassette_data = cassette_data.rename(columns={'k': 'subject', 'sex (F=1)': 'sex', 'LightsOff': 'lights_off'}) # rename columns to be machine readable
cassette_data['sex'] = cassette_data['sex'] - 1

# Circularize time
time_encoder = CircularEncoder()

cassette_data['lights_off'] = cassette_data['lights_off'].apply(lambda t: (t.hour + t.minute/60)/24)
cassette_data['lights_off_cos'], cassette_data['lights_off_sin'] = time_encoder.transform(cassette_data['lights_off'])
del cassette_data['lights_off']

# Get PSG filenames
cassette_psgs = [s for s in os.listdir(cassette_path) if 'PSG' in s]
cassette_psgs.sort(key=lambda s: int(s[3:6]))
cassette_data['psg_filename'] = cassette_psgs

# Get hypnogram filenames
cassette_hypnograms = [s for s in os.listdir(cassette_path) if 'Hypnogram' in s]
cassette_hypnograms.sort(key=lambda s: int(s[3:6]))
cassette_data['hypnogram_filename'] = cassette_hypnograms

# Get technicians
cassette_data['technician'] = [s[7] for s in cassette_data['hypnogram_filename']]

# Add EEG signal data
new_cassette_data = []
for night_idx, base_row in cassette_data.iterrows():
    # Unpack and repack features
    psg_filename = base_row['psg_filename']
    hypnogram_filename = base_row['hypnogram_filename']
    base_row = base_row.tolist()
    
    prefilter = pyedflib.highlevel.read_edf(os.path.join(cassette_path, psg_filename))[1][0]['prefilter']
    
    # Loop through annotations
    annotations = pyedflib.highlevel.read_edf_header(os.path.join(cassette_path, hypnogram_filename))['annotations']
    
    for annotation in annotations:
        print(annotation)
    
    # DELETE
    signal = pyedflib.highlevel.read_edf(os.path.join(cassette_path, psg_filename))[0][0]
    print(len(signal))

    raise KeyboardInterrupt

# Get EEG signal
# signal = pyedflib.highlevel.read_edf(os.path.join(cassette_path, psg_filename))[0][0]
#
# Sleep waves
# Delta: 0-3.5 Hz
# Theta: 4-7.5 Hz
# Alpha: 8-13 Hz
# Beta: 14-30 Hz
# Gamma 30-100 Hz

# Add indicator for cassette data

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
