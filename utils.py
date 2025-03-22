import time
from collections import deque

import numpy as np
from scipy.fft import rfft, rfftfreq
from sklearn.base import BaseEstimator, TransformerMixin

SAMPLE_FREQ = 100 # Hz, sample frequency of all data

# Compute real DFT of a signal
def ft(signal, sample_freq):
    freqs = rfftfreq(len(signal), 1/sample_freq)
    amps = 2*np.abs(rfft(signal))/len(signal)

    return freqs, amps

# Define a circular encoder for cylical data
class CircularEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, period=1):
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        theta = 2*np.pi*X/self.period
        return np.cos(theta), np.sin(theta)

    def inverse_transform(self, X):
        X = np.asarray(X)
        theta = np.arctan2(X[:, 1], X[:, 0])
        return theta*self.period/(2*np.pi)

# Print status messages for a process
def print_status(status_deque, end_var=0, delay=0.5):
    last_print_time = time.time()

    while True:
        # Get message
        try:
            message = status_deque.pop()
        except IndexError:
            continue
        
        # Check for end condition
        if message == end_var:
            break
        
        # Print message
        current_time = time.time()
        if current_time - last_print_time >= delay:
            print(message, end='\r')
            last_print_time = current_time
