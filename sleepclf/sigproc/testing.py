import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.stats import norm

from sleepclf.config import FS, SIGPROC_SETTINGS, OUTPUT_PATH
from .preprocess import condition_signal

# Generate noise for a signal
def generate_noise(length, amp, conf):
    Z = norm.ppf(1 - (1 - conf)/2)
    sigma = amp/Z
    return np.random.normal(loc=0, scale=sigma, size=length)


# Generate signal with arbitrary frequencies, amplitudes, and noise
def generate_signal(f_A_map, fs, dur=None, n_samples=None,
                    noise_amp=0, noise_conf=0.95):
    if not ((dur is None) ^ (n_samples is None)):
        raise ValueError("You must specify exactly one of 'dur' or 'n_samples'"
                         ", but not both.")

    if dur is not None:
        t = np.linspace(0, dur, int(fs*dur))
    else:
        t = np.linspace(0, n_samples/fs, n_samples)

    n = len(t)
    x = np.zeros(n)

    for f, A in f_A_map.items():
        x += A*np.sin(2*np.pi*f*t)

    x += generate_noise(n, noise_amp, noise_conf)

    return t, x

# Compute real DFT of a signal
def ft(x, fs):
    freqs = rfftfreq(len(x), 1/fs)
    amps = 2*np.abs(rfft(x))/len(x)

    return freqs, amps

# Compares signal with its processed version
def compare_signal(t, x, fs):
    f, A = ft(x, fs)

    x_proc = condition_signal(x, SIGPROC_SETTINGS)
    f_proc, A_proc = ft(x_proc, fs)

    fig, ax = plt.subplots(nrows=2, ncols=2)
    
    ax[0][0].plot(t, x, color='lightgreen')
    ax[0][0].set_title("Unprocessed Signal")
    ax[0][0].set_xlabel('t (s)')
    ax[0][0].set_ylabel('Amplitude')
    ax[0][0]

    ax[1][0].plot(t, x_proc, color='green')
    ax[1][0].set_title("Processed Signal")
    ax[1][0].set_xlabel('t (s)')
    ax[1][0].set_ylabel('Amplitude')

    ax[0][1].plot(f, A, color='lightblue')
    ax[0][1].set_title("Unprocessed Signal Frequency Spectrum")
    ax[0][1].set_xlabel('f (Hz)')
    ax[0][1].set_ylabel('Amplitude')

    ax[1][1].plot(f_proc, A_proc, color='blue')
    ax[1][1].set_title("Processed Signal Frequency Spectrum")
    ax[1][1].set_xlabel('f (Hz)')
    ax[1][1].set_ylabel('Amplitude')

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH / 'signal_comparison.svg', bbox_inches='tight')

if __name__ == '__main__':
    f_A_map = {
            5: 1,
            12: 1
    }
    t, x = generate_signal(f_A_map, FS, n_samples=30,
                           noise_amp=0.1)
    compare_signal(t, x, FS)
