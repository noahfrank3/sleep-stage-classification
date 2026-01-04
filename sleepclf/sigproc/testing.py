import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.stats import norm, chi2

from config.config import CONFIG
from .preprocess import condition_signal

FS = CONFIG['sigproc']['sample frequency']
OUTPUT_PATH = CONFIG['paths']['output']

# Generate noise for a signal
def generate_noise(length, amp, conf):
    Z = norm.ppf(1 - (1 - conf)/2)
    sigma = amp/Z
    return np.random.normal(loc=0, scale=sigma, size=length)


# Generate signal with arbitrary frequencies, amplitudes, and noise
def generate_signal(f_A_map, dur=None, n_samples=None,
                    noise_amp=0, noise_conf=0.95):
    if not ((dur is None) ^ (n_samples is None)):
        raise ValueError("You must specify exactly one of 'dur' or 'n_samples'"
                         ", but not both.")

    if dur is not None:
        t = np.linspace(0, dur, int(FS*dur))
    else:
        t = np.linspace(0, n_samples/FS, n_samples)

    n = len(t)
    x = np.zeros(n)

    for f, A in f_A_map.items():
        x += A*np.sin(2*np.pi*f*t)

    x += generate_noise(n, noise_amp, noise_conf)

    return t, x

# Compute real DFT of a signal
def ft(x):
    freqs = rfftfreq(len(x), 1/FS)
    amps = 2*np.abs(rfft(x))/len(x)

    return freqs, amps

# Compares two signals
def compare_signal(t_1, x_1, t_2, x_2):
    f_1, A_1 = ft(x_1)
    f_2, A_2 = ft(x_2)

    fig, ax = plt.subplots(nrows=2, ncols=2)
    
    ax[0][0].plot(t_1, x_1, color='lightgreen')
    ax[0][0].set_title("Signal 1")
    ax[0][0].set_xlabel('t (s)')
    ax[0][0].set_ylabel('Amplitude')
    ax[0][0]

    ax[1][0].plot(t_2, x_2, color='green')
    ax[1][0].set_title("Signal 2")
    ax[1][0].set_xlabel('t (s)')
    ax[1][0].set_ylabel('Amplitude')

    ax[0][1].plot(f_1, A_1, color='lightblue')
    ax[0][1].set_title("Signal 1 Frequency Spectrum")
    ax[0][1].set_xlabel('f (Hz)')
    ax[0][1].set_ylabel('Amplitude')

    ax[1][1].plot(f_2, A_2, color='blue')
    ax[1][1].set_title("Signal 2 Frequency Spectrum")
    ax[1][1].set_xlabel('f (Hz)')
    ax[1][1].set_ylabel('Amplitude')

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH / 'signal_comparison.svg', bbox_inches='tight')

# 1. Relative amplitudes preserved in same signal
# 2. Amplitudes reduced by the same factor in different signals

def generate_f_A_map(mean_num_freqs, mean_num_samples, max_amp=1):
    num_freqs = np.random.geometric(1/mean_num_freqs)
    num_samples = np.random.geometric(1/mean_num_samples)

    f_A_map = {}
    for i in range(num_freqs):
        f = np.random.uniform(0, FS/2)
        A = np.random.uniform(0, max_amp)
        f_A_map[f] = A
    return f_A_map, num_samples

def val_CI(x, conf=0.95):
    n = len(x)
    avg = np.mean(x)
    std = np.std(x, ddof=1)

    alpha = 1 - conf
    Z = norm.ppf(1 - alpha/2)

    L = avg - Z*std/np.sqrt(n)
    U = avg + Z*std/np.sqrt(n)

    return avg, (L, U)

def std_CI(x, conf=0.95):
    n = len(x)
    alpha = 1 - conf

    chi2_L = chi2.ppf(alpha/2, df=n-1)
    chi2_U = chi2.ppf(1 - alpha/2, df=n-1)

    std = np.std(x, ddof=1)
    L = std*np.sqrt((n - 1)/chi2_U)
    U = std*np.sqrt((n - 1)/chi2_L)

    return std, (L, U)

from scipy.signal import welch

def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simpson
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simpson(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simpson(psd, dx=freq_res)
    return bp

if __name__ == '__main__':
    '''
    f_A_map, num_samples = generate_f_A_map(7.5, 500)
    t_1, x_1 = generate_signal(f_A_map, n_samples=num_samples, noise_amp=0.5)
    x_1 = condition_signal(x_1)
    f_1, A_1 = ft(x_1)

    t_2 = t_1
    t_2, x_2 = generate_signal(f_A_map, n_samples=4*num_samples, noise_amp=0.5)
    x_2 = condition_signal(x_2)
    f_2, A_2 = ft(x_2)

    # percent_drop = 100*A_2/A_1
    # _, (L, U) = std_CI(percent_drop)

    compare_signal(t_1, x_1, t_2, x_2)
    '''

    f_A_map = {5: 1, 12: 1}
    t_1, x_1 = generate_signal(f_A_map, n_samples=30, noise_amp=0.1)
    t_2, x_2 = generate_signal(f_A_map, n_samples=1000, noise_amp=0.1)

    from .features import power
    power_1 = bandpower(x_1, FS, [4, 6])
    power_2 = bandpower(x_2, FS, [4, 6])
    print(power_1)
    print(power_2)
    compare_signal(t_1, x_1, t_2, x_2)
