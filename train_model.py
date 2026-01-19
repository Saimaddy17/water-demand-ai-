import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("water_data.csv")

# Convert date
data["date"] = pd.to_datetime(data["date"])
data["day"] = data["date"].dt.day

# Encode area
le = LabelEncoder()
data["area_encoded"] = le.fit_transform(data["area"])

# Features and target
X = data[["day", "area_encoded"]]
y = data["actual_demand"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict demand
data["predicted_demand"] = model.predict(X)

# Save updated data
data.to_csv("water_data.csv", index=False)

print("✅ Model trained and predictions saved")
