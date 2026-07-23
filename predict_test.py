import joblib
import pandas as pd

# Load model
model = joblib.load("student_model.pkl")

# New student data
student = pd.DataFrame({
    "Study_Hours": [6],
    "Attendance": [90],
    "Previous_Marks": [80],
    "Assignment": [85]
})


# Prediction
prediction = model.predict(student)

print("Predicted Final Score:", round(prediction[0],2))