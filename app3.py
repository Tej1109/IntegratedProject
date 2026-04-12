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
MAX_RUL = 125
ALERT_THRESHOLD = 30
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
@st.cache_resource
def load_artifacts(fd_name):
    features = joblib.load(f"artifacts/features_{fd_name}.pkl")
    scaler = joblib.load(f"artifacts/scaler_{fd_name}.pkl")

    model = LSTMModel(len(features)).to(DEVICE)
    model.load_state_dict(
        torch.load(f"artifacts/model_{fd_name}.pth", map_location=DEVICE)
    )
    model.eval()

    return model, scaler, features

# -----------------------------
# UNCERTAINTY
# -----------------------------
def predict_with_uncertainty(model, x, n_samples=20):
    model.train()
    preds = []

    for _ in range(n_samples):
        with torch.no_grad():
            preds.append(model(x).cpu().numpy()[0][0])

    preds = np.array(preds)
    return preds.mean(), preds.std()

# -----------------------------
# FUZZY (internal use only)
# -----------------------------
def triangular(x, a, b, c):
    return max(min((x-a)/(b-a), (c-x)/(c-b)), 0)

def fuzzy_inference(rul, uncertainty, trend):
    rul_low = triangular(rul, 0, 0, 50)
    rul_med = triangular(rul, 30, 65, 100)
    rul_high = triangular(rul, 80, 125, 125)

    unc_low = triangular(uncertainty, 0, 0, 5)
    unc_high = triangular(uncertainty, 8, 15, 20)

    trend_bad = triangular(-trend, 0, 0.5, 2)

    healthy = min(rul_high, unc_low)
    warning = max(rul_med, unc_high, trend_bad)
    critical = rul_low

    score = (
        healthy * 1.0 +
        warning * 0.5 +
        critical * 0.0
    ) / (healthy + warning + critical + 1e-6)

    return score

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
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Configuration")

fd_name = st.sidebar.selectbox(
    "Select Dataset Model",
    ["FD001", "FD002", "FD003", "FD004"]
)

model, scaler, feature_cols = load_artifacts(fd_name)

# -----------------------------
# LOAD DATA
# -----------------------------
query = f"""
SELECT * FROM engine_data
WHERE fd_type = '{fd_name}'
ORDER BY unit_id, cycle
"""

data = pd.read_sql(query, conn)

if data.empty:
    st.error(f"🚨 No data found for {fd_name}")
    st.stop()

data = data.drop(columns=["id", "created_at"])
data = data.rename(columns={"unit_id": "unit"})

engine_id = st.sidebar.selectbox("Select Engine", data['unit'].unique())
engine_full = data[data['unit'] == engine_id].reset_index(drop=True)

max_cycle = engine_full['cycle'].max()
engine_full['TRUE_RUL'] = max_cycle - engine_full['cycle']

engine_data = scaler.transform(engine_full[feature_cols])

# -----------------------------
# UI
# -----------------------------
st.title("✈️ Intelligent Engine Monitoring System")

col1, col2, col3, col4 = st.columns(4)

rul_display = col1.empty()
health_display = col2.empty()
risk_display = col3.empty()
unc_display = col4.empty()

trend_display = st.empty()
status_display = st.empty()
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

    for i in range(len(engine_data)):

        row = engine_data[i] + np.random.normal(0, 0.01, size=engine_data[i].shape)

        window.append(row)
        if len(window) > SEQ_LEN:
            window.pop(0)

        if len(window) == SEQ_LEN:

            x = np.array(window).reshape(1, SEQ_LEN, len(feature_cols))
            x_tensor = torch.tensor(x, dtype=torch.float32).to(DEVICE)

            mean_rul, std_rul = predict_with_uncertainty(model, x_tensor)
            pred_rul = mean_rul
            true_rul = engine_full['TRUE_RUL'].iloc[i]

            pred_history.append(pred_rul)
            true_history.append(true_rul)

            pred_history = pred_history[-200:]
            true_history = true_history[-200:]

            # TREND
            trend = 0
            if len(pred_history) > 5:
                trend = pred_history[-1] - pred_history[-5]

            # FUZZY (internal)
            risk_score = fuzzy_inference(pred_rul, std_rul, trend)

            # METRICS
            health = max(0, min(100, (pred_rul / MAX_RUL) * 100))
            risk = (1 - risk_score) * 100

            # GRAPH
            chart_data = pd.DataFrame({
                "Predicted RUL": pred_history,
                "Smoothed": pd.Series(pred_history).rolling(5).mean(),
                "True RUL": true_history
            })

            chart_placeholder.line_chart(chart_data)

            # UI
            rul_display.metric("Predicted RUL", f"{pred_rul:.2f}")
            health_display.metric("Health %", f"{health:.1f}%")
            risk_display.metric("Risk %", f"{risk:.1f}%")
            unc_display.metric("Uncertainty", f"{std_rul:.2f}")

            if trend < 0:
                trend_display.error(f"📉 Degrading ({trend:.2f})")
            else:
                trend_display.success(f"📈 Stable ({trend:.2f})")

            progress_bar.progress(int(health))

            # STATUS
            if risk > 70:
                status_display.error("🚨 HIGH RISK: Failure likely soon")
            elif risk > 40:
                status_display.warning("⚠️ MODERATE RISK")
            else:
                status_display.success("✅ LOW RISK")

            # -----------------------------
            # ALERT SYSTEM (CORRECT)
            # -----------------------------
            if pred_rul < ALERT_THRESHOLD:

                cursor.execute(
                    """
                    SELECT 1 FROM maintenance_alerts 
                    WHERE unit_id=%s AND fd_type=%s AND alert_status='active'
                    """,
                    (int(engine_id), fd_name)
                )

                if not cursor.fetchone():
                    cursor.execute(
                        """
                        INSERT INTO maintenance_alerts
                        (unit_id, fd_type, alert_triggered_at, predicted_rul_at_alert, threshold_used, severity_level)
                        VALUES (%s, %s, clock_timestamp(), %s, %s, %s)
                        """,
                        (
                            int(engine_id),
                            fd_name,
                            float(pred_rul),
                            ALERT_THRESHOLD,
                            "HIGH"
                        )
                    )

            # STORE PREDICTIONS
            cursor.execute(
                """
                INSERT INTO predictions 
                (unit_id, fd_type, predicted_rul, actual_rul, uncertainty, health_score, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, clock_timestamp())
                """,
                (
                    int(engine_id),
                    fd_name,
                    float(pred_rul),
                    float(true_rul),
                    float(std_rul),
                    float(health)
                )
            )

        time.sleep(0.05)

    conn.commit()

# -----------------------------
# DATABASE TABLES
# -----------------------------
st.divider()
st.subheader("📂 Database Tables")

# Latest
st.markdown("### ⚡ Latest Prediction Per Engine")

latest_df = pd.read_sql(
    f"""
    SELECT DISTINCT ON (unit_id)
        unit_id,
        fd_type,
        predicted_rul,
        actual_rul,
        uncertainty,
        health_score,
        ABS(predicted_rul - actual_rul) AS error,
        created_at
    FROM predictions
    WHERE fd_type = '{fd_name}'
    ORDER BY unit_id, created_at DESC
    """,
    conn
)

st.dataframe(latest_df, use_container_width=True)

# History
st.markdown("### 📊 Prediction History")

pred_df = pd.read_sql(
    f"""
    SELECT unit_id, fd_type, predicted_rul, actual_rul,
           uncertainty,
           health_score,
           ABS(predicted_rul - actual_rul) AS error,
           created_at
    FROM predictions
    WHERE fd_type = '{fd_name}'
    ORDER BY created_at DESC
    LIMIT 50
    """,
    conn
)

st.dataframe(pred_df, use_container_width=True)

# ACTIVE ALERTS (FIXED)
st.markdown("### 🚨 Active Alerts")

alerts_df = pd.read_sql(
    f"""
    SELECT unit_id, predicted_rul_at_alert, threshold_used, severity_level, alert_triggered_at
    FROM maintenance_alerts
    WHERE fd_type = '{fd_name}' AND alert_status = 'active'
    ORDER BY alert_triggered_at DESC
    """,
    conn
)

if alerts_df.empty:
    st.success("No active alerts 🎉")
else:
    st.dataframe(alerts_df, use_container_width=True)

# -----------------------------
# INFO
# -----------------------------
st.caption("""
Interpretation:
- Health % = Remaining useful life
- Risk % = Failure likelihood (fuzzy logic)
- Alerts = Persistent threshold-based warnings
""")