"""
Smart AQI Monitoring & Prediction System
=========================================
Dataset : Global Air Pollution Dataset (Kaggle – hasibalmuzdadid)
Features: CO, Ozone, NO2, PM2.5  |  NOTE: PM10 absent in dataset
Model   : RandomForestRegressor
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import joblib
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart AQI Monitor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2a2d3e);
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 4px solid;
        margin-bottom: 8px;
    }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
FEATURE_COLS = ["CO AQI Value", "Ozone AQI Value", "NO2 AQI Value", "PM2.5 AQI Value"]
TARGET_COL   = "AQI Value"
MODEL_PATH   = "aqi_model.pkl"

AQI_THRESHOLDS = [
    (0,   50,  "Good",                          "#00E400"),
    (51,  100, "Moderate",                       "#FFFF00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#FF7E00"),
    (151, 200, "Unhealthy",                      "#FF0000"),
    (201, 300, "Very Unhealthy",                 "#8F3F97"),
    (301, 999, "Hazardous",                      "#7E0023"),
]

HEALTH_TIPS = {
    "Good":                          "✅ Air quality is satisfactory. Great day for outdoor activities!",
    "Moderate":                      "😐 Acceptable air quality. Unusually sensitive people should limit prolonged exertion.",
    "Unhealthy for Sensitive Groups":"⚠️ Children, elderly & those with respiratory issues should reduce outdoor time.",
    "Unhealthy":                     "🚨 Everyone may experience effects. Limit prolonged outdoor exertion.",
    "Very Unhealthy":                "🔴 Health alert! Everyone should avoid outdoor activities.",
    "Hazardous":                     "☠️ Emergency conditions! Stay indoors, seal windows. Avoid ALL outdoor exposure.",
}


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def get_aqi_info(aqi_val: float):
    for lo, hi, cat, color in AQI_THRESHOLDS:
        if lo <= aqi_val <= hi:
            return cat, color
    return "Hazardous", "#7E0023"


def send_notification(aqi_val: float, location: str, threshold: int):
    """
    Fires both:
      1. Streamlit toast  (always works)
      2. Desktop popup via plyer (works on Windows / Mac / Linux with notify-send)
    """
    if aqi_val <= threshold:
        return False

    cat, _ = get_aqi_info(aqi_val)
    title   = "⚠️  AQI Health Alert"
    message = f"{location} — AQI {aqi_val:.0f} ({cat}). Take precautions!"

    # ── In-app toast ──
    st.toast(f"🚨 {message}", icon="🔔")

    # ── Desktop notification (plyer) ──
    try:
        from plyer import notification as dn
        dn.notify(
            title=title,
            message=message,
            app_name="Smart AQI Monitor",
            timeout=8,
        )
    except Exception:
        pass   # plyer may not be available in all deployment environments

    return True


# ──────────────────────────────────────────────────────────────
# DATA LOADING
@st.cache_data(show_spinner="Loading dataset…")
def load_data(path: str = "global air pollution dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    
    # Clean column names
    df.columns = df.columns.str.strip()

    # 🔥 ADD THIS HERE (IMPORTANT FIX)
    df["Country"] = df["Country"].fillna("Unknown").astype(str)
    df["City"] = df["City"].fillna("Unknown").astype(str)

    # Drop rows with missing feature values
    df.dropna(subset=FEATURE_COLS + [TARGET_COL], inplace=True)

    return df
# ──────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model…")
def get_model(df: pd.DataFrame) -> RandomForestRegressor:
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    joblib.dump(model, MODEL_PATH)
    return model


def model_metrics(model, df):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_te)
    return mean_absolute_error(y_te, y_pred), r2_score(y_te, y_pred)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    # ── Header ──────────────────────────────────────────────
    st.title("🌫️  Smart Air Quality Index Monitor")
    st.caption(
        "ML-powered AQI Prediction · Health Alerts · Trend Analysis  |  "
        "Dataset: [Global Air Pollution – Kaggle](https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset)"
    )

    # ── Load data ────────────────────────────────────────────
    try:
        df = load_data()
    except FileNotFoundError:
        st.error(
            "❌ **Dataset not found.**\n\n"
            
        )
        st.stop()

    model = get_model(df)

    # ── SIDEBAR ──────────────────────────────────────────────
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/wind.png", width=64)
        st.header("⚙️  Filters")

        countries = ["All"] + sorted(df["Country"].dropna().astype(str).unique())
        sel_country = st.selectbox("🌍 Country", countries)

        city_pool = (
            df["City"].unique() if sel_country == "All"
            else df[df["Country"] == sel_country]["City"].dropna().astype(str).unique()
        )
        sel_city = st.selectbox("🏙️ City", ["All"] + sorted(city_pool))

        st.divider()

        st.subheader("🔔  Notification Settings")
        notif_threshold = st.slider(
            "Alert me when AQI exceeds:", 50, 300, 100, step=10,
            help="Triggers both in-app toast and a desktop popup via plyer."
        )
        notif_enabled = st.toggle("Enable Notifications", value=True)

        st.divider()
        st.caption("⚠️ **PM10 Note:** The Kaggle dataset does not include PM10 data. "
                   "Prediction uses CO, Ozone, NO₂, and PM2.5 only.")

    # ── Filter ───────────────────────────────────────────────
    mask = pd.Series([True] * len(df), index=df.index)
    if sel_country != "All":
        mask &= df["Country"] == sel_country
    if sel_city != "All":
        mask &= df["City"] == sel_city
    filtered = df[mask]

    location_label = (
        sel_city if sel_city != "All"
        else sel_country if sel_country != "All"
        else "Global"
    )

    # ──────────────────────────────────────────────────────────
    # TABS
    # ──────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊  Overview", "📈  Trend Analysis", "🤖  AQI Prediction"])

    # ═══════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ═══════════════════════════════════════════════════
    with tab1:
        if filtered.empty:
            st.warning("No data available for this selection.")
        else:
            avg_aqi = filtered[TARGET_COL].mean()
            max_aqi = filtered[TARGET_COL].max()
            min_aqi = filtered[TARGET_COL].min()
            n_cities = filtered["City"].nunique()
            cat, color = get_aqi_info(avg_aqi)

            # Notification check on live overview AQI
            if notif_enabled:
                if send_notification(avg_aqi, location_label, notif_threshold):
                    st.warning(
                        f"🔔 **Notification Fired!** Average AQI ({avg_aqi:.1f}) "
                        f"exceeds your threshold of {notif_threshold}."
                    )

            # KPI cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg AQI",  f"{avg_aqi:.1f}", cat)
            c2.metric("Peak AQI", f"{max_aqi:.0f}")
            c3.metric("Min AQI",  f"{min_aqi:.0f}")
            c4.metric("Cities",   n_cities)

            # AQI status badge
            tip = HEALTH_TIPS.get(cat, "")
            st.markdown(
                f"<div style='background:{color}22; border-left:5px solid {color}; "
                f"padding:12px 18px; border-radius:8px; margin:8px 0'>"
                f"<b style='color:{color}; font-size:18px'>{cat}</b> — {tip}</div>",
                unsafe_allow_html=True,
            )

            st.divider()

            # Top polluted cities bar chart
            top_cities = (
                filtered.groupby("City")[TARGET_COL]
                .mean().reset_index()
                .sort_values(TARGET_COL, ascending=False)
                .head(20)
            )
            fig_bar = px.bar(
                top_cities, x=TARGET_COL, y="City", orientation="h",
                color=TARGET_COL, color_continuous_scale="RdYlGn_r",
                title=f"Top Polluted Cities — {location_label}",
                labels={TARGET_COL: "Avg AQI"},
            )
            fig_bar.update_layout(yaxis=dict(autorange="reversed"), height=500)
            st.plotly_chart(fig_bar, use_container_width=True)

            # AQI category pie
            cat_df = filtered["AQI Category"].value_counts().reset_index()
            cat_df.columns = ["Category", "Count"]
            fig_pie = px.pie(
                cat_df, names="Category", values="Count",
                title="AQI Category Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.4,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # ═══════════════════════════════════════════════════
    # TAB 2 — TREND ANALYSIS
    # ═══════════════════════════════════════════════════
    with tab2:
        st.subheader("Pollutant Analysis & Correlations")
        st.info(
            "📌 **PM10 is not present in this dataset.** "
            "Analysis covers CO, Ozone (O₃), NO₂, and PM2.5."
        )

        if filtered.empty:
            st.warning("No data available for this selection.")
        else:
            # Correlation heatmap
            corr = filtered[FEATURE_COLS + [TARGET_COL]].corr()
            fig_heat = px.imshow(
                corr, text_auto=".2f", aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Correlation Matrix: Pollutants vs AQI",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            # Scatter grid
            col_a, col_b = st.columns(2)
            with col_a:
                fig_s1 = px.scatter(
                    filtered, x="PM2.5 AQI Value", y=TARGET_COL,
                    color="AQI Category", opacity=0.55,
                    title="PM2.5 vs Overall AQI",
                    trendline="ols",
                )
                st.plotly_chart(fig_s1, use_container_width=True)
            with col_b:
                fig_s2 = px.scatter(
                    filtered, x="Ozone AQI Value", y=TARGET_COL,
                    color="AQI Category", opacity=0.55,
                    title="Ozone vs Overall AQI",
                    trendline="ols",
                )
                st.plotly_chart(fig_s2, use_container_width=True)

            col_c, col_d = st.columns(2)
            with col_c:
                fig_s3 = px.scatter(
                    filtered, x="CO AQI Value", y=TARGET_COL,
                    color="AQI Category", opacity=0.55,
                    title="CO vs Overall AQI",
                )
                st.plotly_chart(fig_s3, use_container_width=True)
            with col_d:
                fig_s4 = px.scatter(
                    filtered, x="NO2 AQI Value", y=TARGET_COL,
                    color="AQI Category", opacity=0.55,
                    title="NO₂ vs Overall AQI",
                )
                st.plotly_chart(fig_s4, use_container_width=True)

            # Feature importance
            fi_df = pd.DataFrame({
                "Feature": FEATURE_COLS,
                "Importance": model.feature_importances_,
            }).sort_values("Importance")

            fig_fi = px.bar(
                fi_df, x="Importance", y="Feature", orientation="h",
                title="🤖 ML Feature Importance (RandomForest)",
                color="Importance", color_continuous_scale="Blues",
            )
            st.plotly_chart(fig_fi, use_container_width=True)

    # ═══════════════════════════════════════════════════
    # TAB 3 — PREDICTION
    # ═══════════════════════════════════════════════════
    with tab3:
        st.subheader("🤖 Predict AQI from Pollutant Values")
        st.caption("Enter individual pollutant AQI values to get an ML-predicted overall AQI.")

        col1, col2 = st.columns(2)
        with col1:
            co_val    = st.number_input("CO AQI Value",    0, 500, 50,  step=1)
            ozone_val = st.number_input("Ozone AQI Value", 0, 500, 40,  step=1)
        with col2:
            no2_val   = st.number_input("NO₂ AQI Value",   0, 500, 30,  step=1)
            pm25_val  = st.number_input("PM2.5 AQI Value", 0, 500, 60,  step=1)

        st.warning("⚠️ PM10 AQI Value is **excluded** — not present in the Kaggle dataset.")

        if st.button("🔮  Predict AQI", type="primary", use_container_width=True):
            pred = model.predict(np.array([[co_val, ozone_val, no2_val, pm25_val]]))[0]
            cat, color = get_aqi_info(pred)
            tip = HEALTH_TIPS.get(cat, "")

            # Result card
            st.markdown(
                f"""<div style='padding:24px; border-radius:14px;
                    background:{color}22; border-left:6px solid {color}; margin-top:12px'>
                    <h2 style='margin:0; color:{color}'>Predicted AQI: {pred:.1f}</h2>
                    <p style='margin:6px 0 0 0; font-size:17px'><b>Category:</b> {cat}</p>
                    <p style='margin:6px 0 0 0; font-size:15px'>{tip}</p>
                </div>""",
                unsafe_allow_html=True,
            )

            st.divider()

            # Gauge chart
            fig_gauge = {
                "data": [{
                    "type": "indicator", "mode": "gauge+number",
                    "value": pred,
                    "title": {"text": "AQI Level"},
                    "gauge": {
                        "axis": {"range": [0, 300]},
                        "bar":  {"color": color},
                        "steps": [
                            {"range": [0,   50],  "color": "#00E40033"},
                            {"range": [51,  100], "color": "#FFFF0033"},
                            {"range": [101, 150], "color": "#FF7E0033"},
                            {"range": [151, 200], "color": "#FF000033"},
                            {"range": [201, 300], "color": "#8F3F9733"},
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 3},
                            "thickness": 0.75,
                            "value": pred,
                        },
                    },
                }],
                "layout": {"height": 280, "margin": {"t": 40, "b": 10}},
            }
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Notification
            if notif_enabled:
                fired = send_notification(pred, "Custom Prediction", notif_threshold)
                if fired:
                    st.error(
                        f"🔔 **Alert Triggered!** Predicted AQI ({pred:.1f}) exceeds "
                        f"your threshold ({notif_threshold}). Desktop notification sent."
                    )
                else:
                    st.success(
                        f"✅ AQI ({pred:.1f}) is within your safe threshold ({notif_threshold})."
                    )

        # Model performance expander
        with st.expander("📉 Model Performance on Test Set"):
            mae, r2 = model_metrics(model, df)
            m1, m2 = st.columns(2)
            m1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
            m2.metric("R² Score",                  f"{r2:.4f}")
            st.caption(
                "Model: RandomForestRegressor (150 trees) | "
                "Train-test split: 80/20 | Features: CO, Ozone, NO₂, PM2.5"
            )


if __name__ == "__main__":
    main()
