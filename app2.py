import streamlit as st
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import joblib
import psycopg2

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
# DB CONNECTION
# -----------------------------
conn = psycopg2.connect(
    dbname="integrated_project",
    user="postgres",
    password="root",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# -----------------------------
# LOAD DATA (FD001 ONLY)
# -----------------------------
data = pd.read_sql(
    """
    SELECT * FROM engine_data 
    WHERE fd_type = 'FD001'
    ORDER BY unit_id, cycle
    """,
    conn
)

if data.empty:
    st.error("🚨 No FD001 data found in database")
    st.stop()

data = data.drop(columns=["id", "created_at"])
data = data.rename(columns={"unit_id": "unit"})

# Sidebar
engine_id = st.sidebar.selectbox("Select Engine", data['unit'].unique())
engine_full = data[data['unit'] == engine_id].reset_index(drop=True)

# TRUE RUL
max_cycle = engine_full['cycle'].max()
engine_full['TRUE_RUL'] = max_cycle - engine_full['cycle']

# Features
engine_data = scaler.transform(engine_full[feature_cols])

# -----------------------------
# UI
# -----------------------------
st.title("✈️ Engine Predictive Maintenance Dashboard (FD001)")

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
    alert_triggered = False

    for i in range(len(engine_data)):
        row = engine_data[i] + np.random.normal(0, 0.01, size=engine_data[i].shape)

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

            pred_history = pred_history[-200:]
            true_history = true_history[-200:]

            smoothed = pd.Series(pred_history).rolling(5).mean()

            chart_data = pd.DataFrame({
                "Cycle": range(len(pred_history)),
                "Predicted RUL": pred_history,
                "Smoothed": smoothed,
                "True RUL": true_history
            }).set_index("Cycle")

            chart_placeholder.line_chart(chart_data)

            # METRICS
            rul_display.metric("Predicted RUL", f"{pred_rul:.2f}")
            countdown_display.metric("Cycles to Failure", f"{int(pred_rul)}")

            health = max(0, min(100, (pred_rul / 125) * 100))
            health_display.metric("Health %", f"{health:.1f}%")

            progress_bar.progress(int(health))

            # TREND
            if prev_rul is not None:
                delta = pred_rul - prev_rul
                if delta < 0:
                    delta_display.error(f"📉 Degrading ({delta:.2f})")
                else:
                    delta_display.success(f"📈 Improving ({delta:.2f})")

            prev_rul = pred_rul

            # STORE PREDICTION (FD001 TAGGED)
            cursor.execute(
                """
                INSERT INTO predictions 
                (unit_id, fd_type, predicted_rul, actual_rul, created_at)
                VALUES (%s, %s, %s, %s, clock_timestamp())
                """,
                (int(engine_id), "FD001", float(pred_rul), float(true_rul))
            )

            # ALERT SYSTEM (FD001 ONLY)
            if pred_rul < ALERT_THRESHOLD:
                status_display.error("🚨 CRITICAL: Immediate maintenance required!")

                if not alert_triggered:
                    cursor.execute(
                        """
                        SELECT 1 FROM maintenance_alerts 
                        WHERE unit_id=%s AND fd_type='FD001' AND alert_status='active'
                        """,
                        (int(engine_id),)
                    )

                    if not cursor.fetchone():
                        cursor.execute(
                            """
                            INSERT INTO maintenance_alerts
                            (unit_id, fd_type, alert_triggered_at, predicted_rul_at_alert, threshold_used, severity_level)
                            VALUES (%s, %s, clock_timestamp(), %s, %s, %s)
                            """,
                            (int(engine_id), "FD001", float(pred_rul), ALERT_THRESHOLD, "HIGH")
                        )

                    alert_triggered = True

            elif pred_rul < 50:
                status_display.warning("⚠️ Warning: Engine degrading")
            else:
                status_display.success("✅ Engine operating normally")

        time.sleep(0.05)

    conn.commit()

# ==============================
# DATABASE VIEW (FD001 ONLY)
# ==============================
st.divider()
st.subheader("📂 Database Tables")

# Latest
st.markdown("### ⚡ Latest Prediction Per Engine")

latest_df = pd.read_sql(
    """
    SELECT DISTINCT ON (unit_id)
        unit_id,
        predicted_rul,
        actual_rul,
        ABS(predicted_rul - actual_rul) AS error,
        created_at
    FROM predictions
    WHERE fd_type = 'FD001'
    ORDER BY unit_id, created_at DESC
    """,
    conn
)

st.dataframe(latest_df, use_container_width=True)

# History
st.markdown("### 📊 Prediction History")

total_rows = pd.read_sql(
    "SELECT COUNT(*) FROM predictions WHERE fd_type='FD001'",
    conn
).iloc[0,0]

page_size = 20
max_page = max(1, total_rows // page_size)

page = st.slider("Page", 1, max_page, 1)
offset = (page - 1) * page_size

pred_df = pd.read_sql(
    f"""
    SELECT unit_id, predicted_rul, actual_rul,
           ABS(predicted_rul - actual_rul) AS error,
           created_at
    FROM predictions
    WHERE fd_type = 'FD001'
    ORDER BY created_at DESC
    LIMIT {page_size} OFFSET {offset}
    """,
    conn
)

st.dataframe(pred_df, use_container_width=True)

# Alerts
st.markdown("### 🚨 Active Alerts")

alerts_df = pd.read_sql(
    """
    SELECT unit_id, predicted_rul_at_alert, severity_level, alert_triggered_at
    FROM maintenance_alerts
    WHERE fd_type = 'FD001' AND alert_status = 'active'
    ORDER BY alert_triggered_at DESC
    """,
    conn
)

if alerts_df.empty:
    st.success("No active alerts 🎉")
else:
    st.dataframe(alerts_df, use_container_width=True)