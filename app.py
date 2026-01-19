import streamlit as st
import pandas as pd

st.set_page_config(page_title="Water Demand AI", layout="centered")

st.title("💧 AI-Based Water Demand Forecasting")
st.write("Campus Water Management System")

# Load data
data = pd.read_csv("water_data.csv")

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

# -------- RAG LOGIC --------
st.subheader("🤖 AI Recommendation (RAG)")

average = filtered["actual_demand"].mean()

if latest["predicted_demand"] > average:
    st.warning("High demand expected. Start pumps early and store extra water.")
elif latest["predicted_demand"] < average:
    st.success("Low demand expected. Reduce pumping to save electricity.")
else:
    st.info("Normal demand expected. Follow standard schedule.")
