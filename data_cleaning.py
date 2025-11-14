import pandas as pd
import numpy as np

# Load
df = pd.read_csv("student.csv")
print('Initial shape:', df.shape)

# Drop duplicates
df.drop_duplicates(inplace=True)

# Check missing values and fill if any
print('Missing values before:\n', df.isnull().sum())

# Fill missing values with median for each column
for col in ['hours_studied', 'attendance', 'previous_score', 'final_score']:
    df[col] = df[col].fillna(df[col].median())

# Remove unrealistic rows
df = df[
    (df['hours_studied'] >= 0) &
    (df['attendance'] >= 0) &
    (df['attendance'] <= 100)
]

# Optional: create new feature
df['study_efficiency'] = df['hours_studied'] * (df['previous_score'] / 100)

# Save cleaned data
df.to_csv("cleaned_student.csv", index=False)
print('Cleaned shape:', df.shape)
print('Saved -> data/cleaned_students.csv')
