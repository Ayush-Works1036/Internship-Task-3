import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Predictive Analytics Dashboard", page_icon="📈", layout="wide")
st.title("📈 Predictive Analytics Using Historical Data")
st.write("Forecast future sales trends from historical data using Linear Regression.")

@st.cache_data
def load_data():
    d = pd.read_csv("historical_sales.csv")
    d["Date"] = pd.to_datetime(d["Date"])
    return d

df = load_data()
st.sidebar.header("Forecast Settings")
forecast_days = st.sidebar.slider("Future days to forecast", 7, 90, 30)
test_percent = st.sidebar.slider("Test data percentage", 10, 30, 20, 5)

data = df.copy()
data["Days"] = (data["Date"] - data["Date"].min()).dt.days
X, y = data[["Days"]], data["Sales"]
split = int(len(data) * (1 - test_percent/100))

model = LinearRegression()
model.fit(X.iloc[:split], y.iloc[:split])
pred_test = model.predict(X.iloc[split:])

mae = mean_absolute_error(y.iloc[split:], pred_test)
rmse = np.sqrt(mean_squared_error(y.iloc[split:], pred_test))
r2 = r2_score(y.iloc[split:], pred_test)

future_days = np.arange(data["Days"].max()+1, data["Days"].max()+forecast_days+1)
future_dates = pd.date_range(data["Date"].max()+pd.Timedelta(days=1), periods=forecast_days)
future_pred = model.predict(pd.DataFrame({"Days": future_days}))
forecast = pd.DataFrame({"Date": future_dates, "Predicted_Sales": future_pred.round(2)})

c1,c2,c3,c4 = st.columns(4)
c1.metric("Historical Records", f"{len(data):,}")
c2.metric("Average Sales", f"{data.Sales.mean():.2f}")
c3.metric("R² Score", f"{r2:.3f}")
c4.metric("Forecast Period", f"{forecast_days} days")
st.divider()

st.subheader("📊 Historical Sales Trend")
fig, ax = plt.subplots(figsize=(11,4))
ax.plot(data.Date, data.Sales)
ax.set_xlabel("Date"); ax.set_ylabel("Sales"); ax.grid(alpha=.25)
st.pyplot(fig, use_container_width=True)

st.subheader("🎯 Actual vs Predicted Test Data")
test = data.iloc[split:].copy()
fig, ax = plt.subplots(figsize=(11,4))
ax.plot(test.Date, test.Sales, label="Actual")
ax.plot(test.Date, pred_test, label="Predicted")
ax.set_xlabel("Date"); ax.set_ylabel("Sales"); ax.legend(); ax.grid(alpha=.25)
st.pyplot(fig, use_container_width=True)

st.subheader("📏 Model Accuracy")
a,b,c = st.columns(3)
a.metric("MAE", f"{mae:.2f}")
b.metric("RMSE", f"{rmse:.2f}")
c.metric("R²", f"{r2:.3f}")

st.subheader("🔮 Future Sales Forecast")
fig, ax = plt.subplots(figsize=(11,4))
recent = data.tail(90)
ax.plot(recent.Date, recent.Sales, label="Recent Historical")
ax.plot(forecast.Date, forecast.Predicted_Sales, label="Forecast", linewidth=2)
ax.set_xlabel("Date"); ax.set_ylabel("Sales"); ax.legend(); ax.grid(alpha=.25)
st.pyplot(fig, use_container_width=True)
st.dataframe(forecast, use_container_width=True)

first, last = forecast.Predicted_Sales.iloc[0], forecast.Predicted_Sales.iloc[-1]
pct = (last-first)/first*100
trend = "upward" if pct > 0 else "downward" if pct < 0 else "stable"
st.subheader("💡 Forecast Insights")
st.write(f"The model indicates an **{trend} trend**. Predicted sales change from **{first:.2f}** to **{last:.2f}**, approximately **{pct:.2f}%**.")

st.subheader("⬇️ Export Forecast")
st.download_button("Download Forecast CSV", forecast.to_csv(index=False).encode("utf-8"),
                   "sales_forecast.csv", "text/csv")
