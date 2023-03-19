from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


samp_freq = 1000  # Sample frequency (Hz)
notch_freq = 50.0  # Frequency to be removed from signal (Hz)
quality_factor = 20.0  # Quality factor

b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)

f1 = 120  # low_cut Frequency of 1st signal in Hz
f2 = 200  # high_cut Frequency of 2nd signal in Hz
# Set time vector
# Generate 1000 sample sequence in 1 sec
samples = 50
order = 2
nyq = 0.5 * samples

low = f1 / nyq
high = f2 / nyq

b,a = signal.butter(order, [low, high], btype='bandstop')

print("b: ", b)
print("a: ", a)

# y = signal.filtfilt(b, a, raw_signal)


