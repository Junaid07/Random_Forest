import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt
import gdown
import os

# === Download Random Forest model from Google Drive if not present ===
model_url = "https://drive.google.com/uc?id=15DfplYlCkQlRzw_iwtT3XNq-7R_GKk49"
model_path = "rf_discharge_model.pkl"
if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# === Load model and scaler ===
rf = joblib.load(model_path)
scaler = joblib.load("rf_scaler.pkl")  # Make sure rf_scaler.pkl exists locally

# === Load historical data ===
df = pd.read_excel("Maralla_Total.xlsx")
df.columns = ["Date", "Discharge"]
df["Date"] = pd.to_datetime(df["Date"], format="%d%m%Y", errors="coerce")
df.dropna(subset=["Date", "Discharge"], inplace=True)
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Year"] = df["Date"].dt.year

# === Streamlit UI ===
st.title("🔮 1-Day Discharge Forecast App (Random Forest)")
st.markdown("Enter **today's** and **yesterday's** discharge to forecast **tomorrow's** discharge.")

# === Forecast Inputs ===
today = st.number_input("Today's Discharge (Day 0)", min_value=0.0, step=100.0)
yesterday = st.number_input("Yesterday's Discharge (Day -1)", min_value=0.0, step=100.0)

if st.button("Forecast Next Day"):
    if today == 0.0 or yesterday == 0.0:
        st.warning("Please enter both values to forecast.")
    else:
        input_scaled = scaler.transform([[yesterday, today]])
        prediction = rf.predict(input_scaled)[0]

        # === Historical average for this calendar day ===
        today_date = datetime.date.today()
        same_day_data = df[(df["Date"].dt.month == today_date.month) & (df["Date"].dt.day == today_date.day)]
        historical_avg = same_day_data["Discharge"].mean()

        # === Display Results ===
        st.success(f"📈 Predicted Discharge for Day +1: **{round(prediction, 2)} cfs**")
        if not np.isnan(historical_avg):
            st.info(f"📊 Historical Avg Discharge on {today_date.strftime('%B %d')}: **{round(historical_avg, 2)} cfs**")
        else:
            st.info("No historical data available for this day.")

# === Monthly Visualization Section ===
st.markdown("## 📊 Monthly Discharge Summary")
month_names = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}
selected_month = st.selectbox("Select a Month", list(month_names.keys()), format_func=lambda x: month_names[x])

# === Daily Averages for Selected Month ===
monthly_data = df[df["Month"] == selected_month]
daily_avg = monthly_data.groupby("Day")["Discharge"].mean()

st.markdown(f"### 📅 Daily Averages for {month_names[selected_month]}")
fig1, ax1 = plt.subplots()
ax1.plot(daily_avg.index, daily_avg.values, marker='o')
ax1.set_title(f"Daily Average Discharge - {month_names[selected_month]}")
ax1.set_xlabel("Day of Month")
ax1.set_ylabel("Discharge (cfs)")
st.pyplot(fig1)

# === Monthly Average (Jan–Dec) ===
monthly_avg = df.groupby("Month")["Discharge"].mean()

st.markdown("### 📆 Monthly Average Discharge (Jan–Dec)")
fig2, ax2 = plt.subplots()
ax2.bar([month_names[m] for m in monthly_avg.index], monthly_avg.values)
ax2.set_ylabel("Average Discharge (cfs)")
ax2.set_xticklabels([month_names[m] for m in monthly_avg.index], rotation=45, ha="right")
st.pyplot(fig2)
