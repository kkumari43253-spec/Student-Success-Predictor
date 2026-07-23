import pandas as pd
import numpy as np

np.random.seed(42)

# Total student records
students = 10000

# Generate features
study_hours = np.random.uniform(0, 10, students)
attendance = np.random.randint(50, 101, students)
previous_marks = np.random.randint(30, 101, students)
assignment = np.random.randint(40, 101, students)

# Small randomness for realistic data
noise = np.random.normal(0, 3, students)

# Calculate final score
final_score = (
    (study_hours * 10 * 0.15) +
    (attendance * 0.30) +
    (previous_marks * 0.30) +
    (assignment * 0.25) +
    noise
)

# Keep score between 0-100
final_score = np.clip(final_score, 0, 100)


# Create dataframe
df = pd.DataFrame({
    "Study_Hours": study_hours.round(1),
    "Attendance": attendance,
    "Previous_Marks": previous_marks,
    "Assignment": assignment,
    "Final_Score": final_score.round(1)
})


# Save dataset
df.to_csv("student_data.csv", index=False)

print("Dataset Created Successfully!")
print(df.head())
print("\nShape:", df.shape)