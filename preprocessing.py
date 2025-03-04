import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

import pyedflib

# Compute real fourier transform
def ft(x, f_s):
    f = rfftfreq(len(x), 1/f_s)
    A = 2*np.abs(rfft(x))/len(x)

    return f, A

# edf extraction example
file = pyedflib.EdfReader('test.edf')
sig = file.readSignal(0)

# Sample plot of a signal
f, A = ft(sig, 100)
plt.plot(f, A)
plt.show()
