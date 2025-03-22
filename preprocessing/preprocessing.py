import pandas as pd

from pathlib import Path
from create_cassette_data import create_cassette_data
from create_telemetry_data import create_telemetry_data

# Create cassette and telemetry data
cassette_data = create_cassette_data()
telemetry_data = create_telemetry_data()

# Save data
path = Path('../data')
X.to_csv(path / 'X.csv', index=False)
y.to_csv(path / 'y.csv', index=False)
y_extras.to_csv(path / 'y_extras.csv', index=False)

# Get EEG signal
# signal = pyedflib.highlevel.read_edf(str(cassette_path / psg_filename))[0][0]
#
# Sleep waves
# Delta: 0-3.5 Hz
# Theta: 4-7.5 Hz
# Alpha: 8-13 Hz
# Beta: 14-30 Hz
# Gamma 30-100 Hz

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
