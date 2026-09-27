import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .emg import GESTURES, generate_emg
from .processing import bandpass_filter, normalize, extract_features

def _feature_from_gesture(gesture, seed, noise=0.035):
    _, raw = generate_emg(
        gesture,
        noise=noise,
        seed=seed,
        fs=1000,
        duration=2.0,
    )
    filtered = bandpass_filter(raw)
    processed = normalize(filtered)
    return extract_features(processed)

def build_training_data(samples_per_class=80):
    X = []
    y = []

    seed = 1000
    for gesture in GESTURES:
        for _ in range(samples_per_class):
            X.append(_feature_from_gesture(gesture, seed))
            y.append(gesture)
            seed += 1

    return np.asarray(X), np.asarray(y)

def train_model():
    X, y = build_training_data()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=42,
        )),
    ])

    model.fit(X, y)
    return model

def predict_gesture(model, feature_vector):
    x = np.asarray(feature_vector, dtype=float).reshape(1, -1)

    prediction = str(model.predict(x)[0])
    probabilities_array = model.predict_proba(x)[0]
    classes = model.named_steps["svm"].classes_

    probabilities = {
        str(label): float(prob)
        for label, prob in zip(classes, probabilities_array)
    }

    # Sort in the project's fixed gesture order for stable UI display.
    probabilities = {
        gesture: probabilities.get(gesture, 0.0)
        for gesture in GESTURES
    }

    return prediction, probabilities
