import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.emg import GESTURES, generate_emg
from src.processing import bandpass_filter, normalize, extract_features, FEATURE_NAMES
from src.model import train_model, predict_gesture
from src.hand import render_hand

st.set_page_config(
    page_title="NeuroGrip AI Simulator",
    page_icon="🦾",
    layout="wide",
)

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.hero {
    padding: 1.2rem 1.5rem;
    border-radius: 18px;
    background: linear-gradient(135deg,#111827,#172554);
    border: 1px solid #334155;
    margin-bottom: 1rem;
}
.hero h1 {margin:0; font-size:2.15rem;}
.hero p {margin:.35rem 0 0; color:#94a3b8;}
.card {
    padding: 1rem;
    border-radius: 16px;
    border: 1px solid #334155;
    background: #0f172a;
}
.metric-box {
    padding: .8rem 1rem;
    border-radius: 12px;
    background: #111827;
    border: 1px solid #334155;
    text-align:center;
}
.small {color:#94a3b8;font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1> NeuroGrip AI Enabled prosthetic Arm </h1>
<p>EMG Gesture Recognition & Virtual Prosthetic Hand Simulator</p>
</div>
""", unsafe_allow_html=True)

# Train once per Streamlit session
@st.cache_resource
def get_model():
    return train_model()

model = get_model()

# ---------- Sidebar ----------
st.sidebar.header("Simulation Controls")
gesture = st.sidebar.selectbox(
    "Select simulated muscle intention",
    GESTURES,
    index=0,
)
noise = st.sidebar.slider(
    "Signal noise",
    min_value=0.0,
    max_value=0.30,
    value=0.10,
    step=0.01,
)
seed = st.sidebar.number_input(
    "Simulation seed",
    min_value=0,
    max_value=999999,
    value=1233,
    step=1,
)

run = st.sidebar.button("▶ Run NeuroGrip", use_container_width=True, type="primary")

# Automatically show a valid initial result.
if "result" not in st.session_state:
    run = True

if run:
    # 1. Generate simulated EMG for the selected intention
    t, raw = generate_emg(gesture, noise=noise, seed=int(seed))

    # 2. Signal processing
    filtered = bandpass_filter(raw, fs=1000.0, lowcut=20.0, highcut=450.0)
    normalized = normalize(filtered)

    # 3. Feature extraction
    feature_vector = extract_features(normalized)

    # 4. ML classification
    prediction, probabilities = predict_gesture(model, feature_vector)

    st.session_state.result = {
        "gesture": gesture,
        "t": t,
        "raw": raw,
        "processed": normalized,
        "features": feature_vector,
        "prediction": prediction,
        "probabilities": probabilities,
    }

result = st.session_state.result

# ---------- Top result ----------
prediction = result["prediction"]
confidence = result["probabilities"].get(prediction, 0.0)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="metric-box"><div class="small">SIMULATED INTENTION</div><h3>{result["gesture"]}</h3></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-box"><div class="small">AI PREDICTION</div><h3>{prediction}</h3></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-box"><div class="small">CONFIDENCE</div><h3>{confidence*100:.1f}%</h3></div>', unsafe_allow_html=True)

st.write("")

left, right = st.columns([1.25, 0.85])

with left:
    st.subheader("EMG Signal Processing")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result["t"], y=result["raw"],
        name="Raw EMG",
        mode="lines",
        line=dict(width=1),
    ))
    fig.add_trace(go.Scatter(
        x=result["t"], y=result["processed"],
        name="Filtered / normalized",
        mode="lines",
        line=dict(width=1.2),
    ))
    fig.update_layout(
        height=390,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        margin=dict(l=10,r=10,t=20,b=10),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Virtual Prosthetic Hand")
    render_hand(prediction)

# ---------- Features + probabilities ----------
st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Extracted Features")
    feature_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Value": [float(x) for x in result["features"]],
    })
    feature_df["Value"] = feature_df["Value"].map(lambda x: f"{x:.5f}")
    st.dataframe(feature_df, use_container_width=True, hide_index=True)

with right:
    st.subheader("AI Classification Probabilities")
    prob_df = pd.DataFrame({
        "Gesture": list(result["probabilities"].keys()),
        "Probability": list(result["probabilities"].values()),
    })
    prob_df["Probability"] *= 100

    fig2 = go.Figure(go.Bar(
        x=prob_df["Probability"],
        y=prob_df["Gesture"],
        orientation="h",
        text=[f"{v:.1f}%" for v in prob_df["Probability"]],
        textposition="auto",
    ))
    fig2.update_layout(
        height=320,
        xaxis_title="Probability (%)",
        yaxis_title="",
        xaxis=dict(range=[0,100]),
        margin=dict(l=10,r=10,t=10,b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("Software Implementation Pipeline")
st.markdown("""
**Simulated EMG → Band-pass filtering → Normalization → Feature extraction → RBF-SVM classification → Virtual hand command**

> **Important:** This version uses synthetic EMG signals for software validation. It does not claim to acquire real EMG hardware data yet. The same processing/classification interface can later accept real sensor samples.
""")
