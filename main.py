
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import lfilter

# Read input audio
try:
    fs, x = wavfile.read("input_audio1.wav")
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(float)
    x = x / max(np.max(np.abs(x)), 1e-10)

except FileNotFoundError:
    fs = 44100
    t0 = np.arange(2 * fs) / fs
    x = np.sin(2 * np.pi * 440 * t0) * np.exp(-4 * t0)
    print("Using generated test tone.")

N = len(x)
t = np.arange(N) / fs

# IIR Echo Filter
D = round(0.25 * fs)
g = 0.55

b = np.zeros(D + 1)
a = np.zeros(D + 1)
b[0] = 1
a[0] = 1
a[D] = -g

echo = lfilter(b, a, x)
echo = echo / max(np.max(np.abs(echo)), 1e-10)

# Four parallel comb filters
delays = [29.7, 37.1, 41.3, 43.7]
gains = [0.82, 0.79, 0.76, 0.73]
comb_sum = np.zeros(N)

for delay, gain in zip(delays, gains):
    D = round(delay * fs / 1000)

    b = np.zeros(D + 1)
    a = np.zeros(D + 1)
    b[0] = 1
    a[0] = 1
    a[D] = -gain

    comb_sum += lfilter(b, a, x)

# Two all-pass filters
reverb = comb_sum

for delay in [5.0, 1.7]:
    D = round(delay * fs / 1000)

    b = np.zeros(D + 1)
    a = np.zeros(D + 1)
    b[0] = -0.7
    b[D] = 1
    a[0] = 1
    a[D] = -0.7

    reverb = lfilter(b, a, reverb)

reverb = reverb / max(np.max(np.abs(reverb)), 1e-10)

# Wet and dry mixing
output = 0.6 * x + 0.4 * reverb
output = output / max(np.max(np.abs(output)), 1e-10)

# Save output audio
wavfile.write("output_echo.wav", fs, echo.astype(np.float32))
wavfile.write("output_reverberation.wav", fs, output.astype(np.float32))

# Plot waveforms
plt.figure(figsize=(10, 7))

plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.title("Original Audio")
plt.ylabel("Amplitude")
plt.xlim(0, min(3, t[-1]))

plt.subplot(3, 1, 2)
plt.plot(t, echo)
plt.title("IIR Echo")
plt.ylabel("Amplitude")
plt.xlim(0, min(3, t[-1]))

plt.subplot(3, 1, 3)
plt.plot(t, output)
plt.title("IIR Reverberation")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.xlim(0, min(3, t[-1]))

plt.tight_layout()
plt.show()

print("Done! Audio files saved.")
