import pandas as pd

# Load dataset
df = pd.read_csv("student_data.csv")

# First 5 rows
print("First 5 Records:")
print(df.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Shape
print("\nDataset Shape:")
print(df.shape)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Correlation
print("\nCorrelation:")
print(df.corr())