import streamlit as st
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import joblib

# -----------------------------
# CONFIG
# -----------------------------
SEQ_LEN = 30
ALERT_THRESHOLD = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# MODEL
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x[:, -1, :]
        return self.fc(x)

# -----------------------------
# LOAD ARTIFACTS
# -----------------------------
feature_cols = joblib.load("features.pkl")
scaler = joblib.load("scaler.pkl")

model = LSTMModel(len(feature_cols)).to(DEVICE)
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.eval()

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("train_FD001.txt", sep=" ", header=None)
data = data.drop(columns=[26, 27])

cols = ['unit', 'cycle'] + [f'op{i}' for i in range(1,4)] + [f's{i}' for i in range(1,22)]
data.columns = cols

# Sidebar
engine_id = st.sidebar.selectbox("Select Engine", data['unit'].unique())
engine_full = data[data['unit'] == engine_id].reset_index(drop=True)

# TRUE RUL (for comparison)
max_cycle = engine_full['cycle'].max()
engine_full['TRUE_RUL'] = max_cycle - engine_full['cycle']

# Features + scaling
engine_data = engine_full[feature_cols]
engine_data = scaler.transform(engine_data)

# -----------------------------
# UI
# -----------------------------
st.title("✈️ Engine Predictive Maintenance Dashboard")

col1, col2, col3 = st.columns(3)

rul_display = col1.empty()
countdown_display = col2.empty()
health_display = col3.empty()

status_display = st.empty()
delta_display = st.empty()
progress_bar = st.progress(0)

chart_placeholder = st.empty()

start = st.button("Start Simulation")

# -----------------------------
# SIMULATION
# -----------------------------
if start:
    window = []
    pred_history = []
    true_history = []

    prev_rul = None

    for i in range(len(engine_data)):
        row = engine_data[i]

        # simulate slight noise
        row = row + np.random.normal(0, 0.01, size=row.shape)

        window.append(row)
        if len(window) > SEQ_LEN:
            window.pop(0)

        if len(window) == SEQ_LEN:
            x = np.array(window).reshape(1, SEQ_LEN, len(feature_cols))
            x_tensor = torch.tensor(x, dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                pred_rul = model(x_tensor).cpu().numpy()[0][0]

            true_rul = engine_full['TRUE_RUL'].iloc[i]

            pred_history.append(pred_rul)
            true_history.append(true_rul)

            # sliding window
            pred_history = pred_history[-200:]
            true_history = true_history[-200:]

            # smoothing
            smoothed = pd.Series(pred_history).rolling(5).mean()

            # chart
            cycles = list(range(len(pred_history)))

            chart_data = pd.DataFrame({
                "Cycle": cycles,
                "Predicted RUL": pred_history,
                "Smoothed": smoothed,
                "True RUL": true_history
            }).set_index("Cycle")

            chart_placeholder.line_chart(chart_data)

            # -----------------------------
            # METRICS
            # -----------------------------
            rul_display.metric("Predicted RUL", f"{pred_rul:.2f}")
            countdown_display.metric("Cycles to Failure", f"{int(pred_rul)}")

            health = max(0, min(100, (pred_rul / 125) * 100))
            health_display.metric("Health %", f"{health:.1f}%")

            # SINGLE progress bar update
            progress_bar.progress(int(health))

            # -----------------------------
            # TREND
            # -----------------------------
            if prev_rul is not None:
                delta = pred_rul - prev_rul
                if delta < 0:
                    delta_display.error(f"📉 Degrading ({delta:.2f})")
                else:
                    delta_display.success(f"📈 Improving ({delta:.2f})")

            prev_rul = pred_rul

            # -----------------------------
            # ALERTS
            # -----------------------------
            if pred_rul < ALERT_THRESHOLD:
                status_display.error("🚨 CRITICAL: Immediate maintenance required!")
            elif pred_rul < 50:
                status_display.warning("⚠️ Warning: Engine degrading")
            else:
                status_display.success("✅ Engine operating normally")

        time.sleep(0.05)