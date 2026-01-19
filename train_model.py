import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Load data
data = pd.read_csv("water_data.csv")

print(data.head())
# Convert date to datetime
data["date"] = pd.to_datetime(data["date"])

# Extract useful time features
data["day"] = data["date"].dt.day
data["month"] = data["date"].dt.month
data["dayofweek"] = data["date"].dt.dayofweek

# Encode area (Hostel, Canteen, etc.)
le = LabelEncoder()
data["area_encoded"] = le.fit_transform(data["area"])
print(data.head())
# Define features and target
X = data[["day", "month", "dayofweek", "area_encoded"]]
y = data["actual_demand"]
print(X.head())
print(y.head())
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize XGBoost model
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)
print("Model trained successfully")
# Generate predictions for all data
data["predicted_demand"] = model.predict(X)

# Save updated data back to CSV
data.to_csv("water_data.csv", index=False)

print("Predictions generated and saved to CSV")
