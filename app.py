import streamlit as st
import pandas as pd

st.title("AI-Based Water Demand Forecasting Dashboard")
st.subheader("Campus Water Management System")

# Load data
data = pd.read_csv("water_data.csv")

st.subheader("Water Usage Data")
st.dataframe(data)

# Area selection
area = st.selectbox("Select Campus Area", data["area"].unique())
filtered_data = data[data["area"] == area]

latest = filtered_data.iloc[-1]

st.metric(
    label="Predicted Water Demand (Liters)",
    value=int(latest["predicted_demand"]),
    delta=int(latest["predicted_demand"] - latest["actual_demand"])
)

st.subheader("Actual vs Predicted Demand")
st.line_chart(
    filtered_data.set_index("date")[["actual_demand", "predicted_demand"]]
)

# ---------- RAG LOGIC ----------
st.subheader("AI Recommendation (RAG-Based Insight)")

historical_avg = filtered_data["actual_demand"].mean()

if latest["predicted_demand"] > historical_avg * 1.1:
    st.warning(
        "High demand predicted. AI suggests starting pumps early and storing extra water."
    )
elif latest["predicted_demand"] < historical_avg * 0.9:
    st.success(
        "Low demand predicted. AI suggests reducing pumping to save electricity."
    )
else:
    st.info(
        "Normal demand predicted. AI suggests following the regular pump schedule."
    )
