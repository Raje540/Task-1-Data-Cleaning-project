# -----------------------------
# Data Cleaning for Titanic Dataset
# -----------------------------

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 2: Load the dataset
data = pd.read_csv("dataset.csv")
print("✅ Dataset Loaded Successfully\n")
print(data.head())

# Step 3: Basic information
print("\n--- Dataset Info ---")
print(data.info())

# Step 4: Check for missing values
print("\n--- Missing Values Before Cleaning ---")
print(data.isnull().sum())

# Step 5: Handle missing values
# Fill Age and Fare with mean
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Fare'].fillna(data['Fare'].mean(), inplace=True)

# Fill Embarked with most common value (mode)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Step 6: Drop unnecessary columns
data.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'], inplace=True)

# Step 7: Convert categorical columns to numeric
encoder = LabelEncoder()
data['Sex'] = encoder.fit_transform(data['Sex'])  # Male=1, Female=0
data['Embarked'] = encoder.fit_transform(data['Embarked'])  # C, Q, S

# Step 8: Scale numerical columns
scaler = StandardScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

# Step 9: Detect and remove outliers using IQR for 'Fare'
Q1 = data['Fare'].quantile(0.25)
Q3 = data['Fare'].quantile(0.75)
IQR = Q3 - Q1
data = data[(data['Fare'] >= Q1 - 1.5 * IQR) & (data['Fare'] <= Q3 + 1.5 * IQR)]

# Step 10: Save the cleaned dataset
data.to_csv("cleaned_dataset.csv", index=False)
print("\n✅ Cleaned dataset saved as 'cleaned_dataset.csv'")

# Step 11: Visualizations
sns.boxplot(x='Pclass', y='Age', data=data)
plt.title("Age Distribution by Passenger Class")
plt.show()

sns.countplot(x='Survived', hue='Sex', data=data)
plt.title("Survival Count by Gender")
plt.show()
