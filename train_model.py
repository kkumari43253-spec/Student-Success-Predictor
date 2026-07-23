import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib


# Load dataset
df = pd.read_csv("student_data.csv")


# Input and Output
X = df.drop("Final_Score", axis=1)
y = df["Final_Score"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# Train
model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)


print("Model Training Completed!")
print("-------------------------")
print("MAE:", round(mae,2))
print("RMSE:", round(rmse,2))
print("R2 Score:", round(r2,4))


# Save model
joblib.dump(model, "student_model.pkl")

print("\nModel Saved Successfully!")