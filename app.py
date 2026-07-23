from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("student_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    study_hours = float(request.form["study_hours"])
    attendance = float(request.form["attendance"])
    previous_marks = float(request.form["previous_marks"])
    assignment = float(request.form["assignment"])


    data = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Previous_Marks": [previous_marks],
        "Assignment": [assignment]
    })


    prediction = model.predict(data)[0]
    prediction = round(prediction,2)


    # Grade
    if prediction >= 85:
        grade = "A"
    elif prediction >= 70:
        grade = "B"
    elif prediction >= 50:
        grade = "C"
    else:
        grade = "D"


    # Performance Level
    if prediction >= 85:
        level = "Excellent"
    elif prediction >= 70:
        level = "Good"
    elif prediction >= 50:
        level = "Average"
    else:
        level = "Needs Improvement"


    # Success Probability
    probability = min(round(prediction + 10, 2), 99)


    # Risk Level + Suggestion
    if prediction >= 70:
        risk = "Low"
        message = "Your performance is good. Maintain your routine and keep improving."
    elif prediction >= 50:
        risk = "Medium"
        message = "Focus more on studies and improve consistency."
    else:
        risk = "High"
        message = "Need immediate improvement in study habits."


    return render_template(
        "index.html",
        prediction=prediction,
        grade=grade,
        level=level,
        probability=probability,
        risk=risk,
        message=message
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)