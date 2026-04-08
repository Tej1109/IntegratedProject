
import streamlit as st
import numpy as np
import pandas as pd
import time
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# CONFIG
# -----------------------------
SEQ_LEN = 30
ALERT_THRESHOLD = 20

# -----------------------------
# LOAD ARTIFACTS
# -----------------------------
model = load_model("model.h5", compile=False)
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("features.pkl")

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("train_FD001.txt", sep=" ", header=None)
data = data.drop(columns=[26, 27])

cols = ['unit', 'cycle'] + [f'op{i}' for i in range(1,4)] + [f's{i}' for i in range(1,22)]
data.columns = cols

# Sidebar
engine_id = st.sidebar.selectbox("Select Engine", data['unit'].unique())
engine_data = data[data['unit'] == engine_id].reset_index(drop=True)

# Keep only trained features
engine_data = engine_data[feature_cols]

# Apply scaling
engine_data = scaler.transform(engine_data)

# -----------------------------
# UI
# -----------------------------
st.title("✈️ Engine Predictive Maintenance Dashboard")

rul_display = st.empty()
status_display = st.empty()
chart_placeholder = st.empty()

start = st.button("Start Simulation")

# -----------------------------
# SIMULATION
# -----------------------------
if start:
    window = []
    rul_history = []

    for i in range(len(engine_data)):
        row = engine_data[i]

        # simulate sensor noise
        row = row + np.random.normal(0, 0.01, size=row.shape)

        window.append(row)

        if len(window) > SEQ_LEN:
            window.pop(0)

        if len(window) == SEQ_LEN:
            x = np.array(window).reshape(1, SEQ_LEN, len(feature_cols))
            pred_rul = model.predict(x, verbose=0)[0][0]

            # store REAL predictions
            rul_history.append(pred_rul)

            # trim history FIRST
            if len(rul_history) > 200:
                rul_history = rul_history[-200:]

            # smoothing AFTER trimming
            smoothed_history = pd.Series(rul_history).rolling(window=5).mean()

            # reset index for safety
            smoothed_history = smoothed_history.reset_index(drop=True)

            # update chart
            chart_data = pd.DataFrame({
                "Raw RUL": rul_history,
                "Smoothed RUL": smoothed_history
            })
            chart_placeholder.line_chart(chart_data)

            # display current value
            rul_display.metric("Predicted RUL", f"{pred_rul:.2f}")

            # alerts
            if pred_rul < ALERT_THRESHOLD:
                status_display.error(f"⚠️ CRITICAL: RUL < {ALERT_THRESHOLD}")
            elif pred_rul < 50:
                status_display.warning("⚠️ Warning: Degrading")
            else:
                status_display.success("✅ Healthy")

        time.sleep(0.1)