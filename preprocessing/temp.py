from signal_data import condition_signal
import numpy as np
import matplotlib.pyplot as plt

fs = 100
f1 = 10
f2 = 20

t = np.arange(0, 1, 1/fs)
x = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)

x_proc = condition_signal(x)

fig, ax = plt.subplots(nrows=2)
ax[0].plot(t, x, color='dodgerblue')
ax[1].plot(t, x_proc, color='dodgerblue')
plt.show()
