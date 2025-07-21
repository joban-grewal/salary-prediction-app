import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('data.csv')

# Display initial info
print("Initial Dataset Info:")
print(df.info())
print("First five rows:")
print(df.head())

# Check for null values
print("\n🔍 Null values:")
print(df.isnull().sum())

# Drop rows with nulls (if any)
df.dropna(inplace=True)

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
print(f"Categorical columns found: {list(categorical_cols)}")

# Dictionary to store label encoders
label_encoders = {}

# Encode categorical variables
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoder for later use
    print(f"Encoded: {col}")

# Display cleaned data
print("\nCleaned Data Preview:")
print(df.head())
print("\nData types after encoding:")
print(df.dtypes)

# Save cleaned data
df.to_csv("cleaned_data.csv", index=False)
print("\nSaved as 'cleaned_data.csv'")

# Save the encoders to file
joblib.dump(label_encoders, "preprocessor.pkl")
print("✅ Saved encoders to 'preprocessor.pkl'")