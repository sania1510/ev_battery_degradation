import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="EV Battery Degradation Risk Dashboard", layout="wide")

st.title("🔋 EV Battery Degradation Risk Prediction")
st.sidebar.header("🔧 Adjust Conditions")

st.sidebar.subheader("EV Conditions")
avg_speed = st.sidebar.slider("Average Speed (kmph)", 10, 120, 40)
charging_cycles = st.sidebar.slider("Charging Cycles", 50, 1000, 300)
fast_charging_ratio = st.sidebar.slider("Fast Charging Ratio (%)", 0, 100, 20)
avg_battery_temp = st.sidebar.slider("Battery Temperature (°C)", 20, 60, 35)

st.sidebar.subheader("Traffic Conditions")
vehicles = st.sidebar.slider("Vehicles per Hour", 100, 5000, 1200)
junction = st.sidebar.selectbox("Junction Type", ["Small", "Medium", "Large"])
weekday = st.sidebar.selectbox("Day Type", ["Weekday", "Weekend"])

st.sidebar.subheader("Weather Conditions")
temperature = st.sidebar.slider("Ambient Temperature (°C)", -10, 50, 25)
humidity = st.sidebar.slider("Humidity (%)", 10, 100, 50)
precip_type = st.sidebar.selectbox("Precipitation", ["None", "Rain", "Snow"])
visibility = st.sidebar.slider("Visibility (km)", 0, 10, 8)

ev_stress = (charging_cycles / 1000) + (fast_charging_ratio / 100) + (avg_battery_temp / 100)
traffic_stress = (vehicles / 5000) + (0.5 if junction == "Large" else 0.3 if junction == "Medium" else 0.1)
traffic_stress += (0.2 if weekday == "Weekday" else 0.1)
weather_stress = (abs(temperature - 25) / 50) + (humidity / 100) + (0.3 if precip_type != "None" else 0) + (1 - visibility / 10)

rri = 0.6 * ev_stress + 0.25 * traffic_stress + 0.15 * weather_stress

st.subheader("📊 Relative Risk Index")
st.metric("Degradation Risk Score", f"{rri:.3f}", help="Higher means more stress on battery health")

if rri < 0.5:
    st.success("✅ Low Risk: Battery health is in good condition under these conditions.")
elif 0.5 <= rri < 1.0:
    st.warning("⚠️ Medium Risk: Monitor usage, avoid frequent fast charging and heavy traffic.")
else:
    st.error("🚨 High Risk: Conditions may accelerate battery degradation significantly.")

st.subheader("📈 Visual Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Degradation Risk Distribution**")
    data = np.random.normal(loc=rri, scale=0.2, size=200)
    fig, ax = plt.subplots()
    sns.histplot(data, kde=True, ax=ax, color="blue")
    ax.axvline(rri, color="red", linestyle="--", label="Your Scenario")
    ax.legend()
    st.pyplot(fig)
with col2:
    st.markdown("**Stress Factor Contribution**")
    df_factors = pd.DataFrame({
        "Factor": ["EV Stress", "Traffic Stress", "Weather Stress"],
        "Value": [ev_stress, traffic_stress, weather_stress]
    })
    fig, ax = plt.subplots()
    sns.barplot(x="Factor", y="Value", data=df_factors, palette="viridis", ax=ax)
    ax.set_title("Relative Contribution of Stress Factors")
    st.pyplot(fig)
st.subheader("📋 Scenario Comparison")
scenarios = pd.DataFrame({
    "Scenario": ["Clear Weather + Low Traffic", "Cloudy + High Traffic", "High Temp + Fast Charging"],
    "Relative Risk Index": [
        0.6 * ev_stress + 0.25 * 0.2 + 0.15 * 0.1,
        0.6 * ev_stress + 0.25 * 0.9 + 0.15 * 0.8,
        0.6 * (ev_stress + 0.5) + 0.25 * traffic_stress + 0.15 * weather_stress
    ]
})
st.table(scenarios)