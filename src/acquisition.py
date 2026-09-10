import numpy as np
from scipy.io import wavfile
import scipy.signal as signal
import matplotlib.pyplot as plt
from project import CMBReconstructor

project = CMBReconstructor(resolution=1024)

samplerate, data = wavfile.read('data/radio static noise.wav')

# Quantifying stereo Correlation

if data.ndim > 1:
    corr_matrix = np.corrcoef(data[:, 0], data[:, 1])
    correlation = corr_matrix[0, 1]
    print(f"Numerical Result - Stereo Channel Correlation = {correlation:.4f}")
    mono_signal = np.mean(data, axis=1)
else:
    mono_signal = data
    print("signal is already mono. No averaging performed")

''' Up there: If the recording is stereo we calculate the correlation between the
two channels and then average them into one signal. If it's already mono, we use it
directly'''

# Convert to floating point before filtering
mono_signal = mono_signal.astype(float)

# Notch filtering the 82.8 Hz hum 
f_notch = 82.8
Q = 30 # Quality factor
w0 = f_notch / (samplerate / 2) # Normalized frequency

b, a = signal.iirnotch(w0, Q)
filtered_signal = signal.filtfilt(b, a, mono_signal)

'''up there: it removes the narrow 82.8 Hz electronic interference while preserving 
the rest of the recorded signal'''

# For the 1024x1024 2D field 
project.raw_1d_signal = filtered_signal[:project.target_samples].astype(float)

'''Takes the first 1048576 samples, the exact amount needed for 1024x1024 field'''

# Plotting the cleaned 1D signal
plt.figure(figsize=(12, 4))
plt.plot(project.raw_1d_signal[:2000], color='darkred', lw=0.7)
plt.title("Notch-Filtered 1D Waveform (82.8 Hz Removed)", fontsize=14)
plt.xlabel("Time(samples)")
plt.ylabel("amplitude")
plt.grid(alpha=0.3)
plt.show()

print(f"Result:{len(project.raw_1d_signal)} ready for 2D folding")
