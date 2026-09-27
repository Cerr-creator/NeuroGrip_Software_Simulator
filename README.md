# NeuroGrip AI — Software Simulator

Software-only prototype for EMG gesture recognition and virtual prosthetic-hand control.

## Pipeline

Synthetic EMG
-> Band-pass filter
-> Normalization
-> Feature extraction
-> RBF-SVM
-> Gesture prediction
-> Virtual hand

## Gestures

- Rest
- Open Hand
- Fist
- Pinch
- Point

## Run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

This prototype uses synthetic EMG for software validation. It does not represent clinical performance and does not yet acquire real sensor data.
