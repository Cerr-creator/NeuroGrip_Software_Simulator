import numpy as np
from scipy.signal import butter, sosfiltfilt

FEATURE_NAMES = [
    "RMS",
    "MAV",
    "Variance",
    "Std Dev",
    "Waveform Length",
    "Zero Crossings",
    "Peak-to-Peak",
]

def bandpass_filter(signal, fs=1000.0, lowcut=20.0, highcut=450.0, order=4):
    x = np.asarray(signal, dtype=float)
    nyq = fs / 2.0
    highcut = min(highcut, nyq - 1.0)
    lowcut = max(lowcut, 1.0)

    sos = butter(
        order,
        [lowcut / nyq, highcut / nyq],
        btype="bandpass",
        output="sos",
    )
    return sosfiltfilt(sos, x)

def normalize(signal):
    x = np.asarray(signal, dtype=float)
    scale = np.max(np.abs(x))
    if scale < 1e-12:
        return x.copy()
    return x / scale

def extract_features(signal):
    x = np.asarray(signal, dtype=float)
    dx = np.diff(x)

    rms = np.sqrt(np.mean(x**2))
    mav = np.mean(np.abs(x))
    variance = np.var(x)
    std = np.std(x)
    waveform_length = np.sum(np.abs(dx))
    zero_crossings = np.sum((x[:-1] * x[1:]) < 0)
    peak_to_peak = np.ptp(x)

    return np.array([
        rms,
        mav,
        variance,
        std,
        waveform_length,
        zero_crossings,
        peak_to_peak,
    ], dtype=float)
