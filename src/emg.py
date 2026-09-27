import numpy as np

GESTURES = ["Rest", "Open Hand", "Fist", "Pinch", "Point"]

# Class-specific signal parameters.
# These are intentionally simple synthetic patterns for software validation,
# not physiological measurements from a patient.
PARAMS = {
    "Rest":      {"amp": 0.12, "freq": 42, "burst": 0.15},
    "Open Hand": {"amp": 0.55, "freq": 72, "burst": 0.70},
    "Fist":      {"amp": 0.90, "freq": 115, "burst": 0.95},
    "Pinch":     {"amp": 0.68, "freq": 165, "burst": 0.62},
    "Point":     {"amp": 0.48, "freq": 225, "burst": 0.52},
}

def _envelope(t, burst):
    # Smooth muscle-activation envelope with two broad activity regions.
    center1 = 0.55
    center2 = 1.25
    width = 0.28
    e1 = np.exp(-0.5 * ((t-center1)/width)**2)
    e2 = np.exp(-0.5 * ((t-center2)/width)**2)
    return 0.15 + burst * (0.65*e1 + 0.35*e2)

def generate_emg(gesture, noise=0.10, seed=1233, fs=1000, duration=2.0):
    if gesture not in GESTURES:
        raise ValueError(f"Unknown gesture: {gesture}")

    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n) / fs
    p = PARAMS[gesture]

    env = _envelope(t, p["burst"])

    # Main class-specific oscillatory components plus harmonics.
    phase = rng.uniform(0, 2*np.pi)
    carrier = (
        np.sin(2*np.pi*p["freq"]*t + phase)
        + 0.45*np.sin(2*np.pi*(p["freq"]*1.7)*t + phase/2)
        + 0.20*np.sin(2*np.pi*(p["freq"]*2.4)*t)
    )

    # Low-amplitude baseline noise and requested observation noise.
    baseline = 0.035 * rng.normal(size=n)
    observation = noise * 0.45 * rng.normal(size=n)

    raw = p["amp"] * env * carrier + baseline + observation

    # Add a class-dependent burst for clearer software-level separability.
    if gesture != "Rest":
        burst_freq = p["freq"] * 0.55
        raw += 0.12 * p["amp"] * env * np.sin(2*np.pi*burst_freq*t + phase)

    return t, raw.astype(float)
