import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Water Demand AI", layout="centered")

st.title("💧 AI-Based Water Demand Forecasting")
st.write("Campus Water Management System")

# Load data
data = pd.read_csv("water_data.csv")

# Feature engineering
data["date"] = pd.to_datetime(data["date"])
data["day"] = data["date"].dt.day

le = LabelEncoder()
data["area_encoded"] = le.fit_transform(data["area"])

X = data[["day", "area_encoded"]]
y = data["actual_demand"]

# Train model INSIDE app (fixes cloud issue)
model = LinearRegression()
model.fit(X, y)

data["predicted_demand"] = model.predict(X)

# UI
area = st.selectbox("Select Area", data["area"].unique())
filtered = data[data["area"] == area]
latest = filtered.iloc[-1]

st.metric(
    "Predicted Water Demand (Liters)",
    int(latest["predicted_demand"]),
    int(latest["predicted_demand"] - latest["actual_demand"])
)

st.line_chart(
    filtered.set_index("date")[["actual_demand", "predicted_demand"]]
)

# RAG-style Recommendation
st.subheader("🤖 AI Recommendation")

avg = filtered["actual_demand"].mean()

if latest["predicted_demand"] > avg:
    st.warning("High demand expected. Start pumps early and store extra water.")
elif latest["predicted_demand"] < avg:
    st.success("Low demand expected. Reduce pumping to save electricity.")
else:
    st.info("Normal demand expected. Follow standard schedule.")
