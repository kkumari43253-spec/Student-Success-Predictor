import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("student_data.csv")


# Correlation Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True)
plt.title("Feature Correlation")
plt.show()


# Final Score Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Final_Score"], bins=30)
plt.title("Student Final Score Distribution")
plt.xlabel("Final Score")
plt.show()


# Features vs Final Score

features = ["Study_Hours", "Attendance", "Previous_Marks", "Assignment"]

for feature in features:
    plt.figure(figsize=(6,4))
    sns.scatterplot(
        x=df[feature],
        y=df["Final_Score"]
    )
    plt.title(f"{feature} vs Final Score")
    plt.show()