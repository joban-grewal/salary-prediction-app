import pandas as pd

df = pd.read_csv('cleaned_data.csv')
print("Initial shape:", df.shape)

# Drop columns not used for modeling if present
cols_to_drop = ['Employee ID', 'Date of Joining', 'Unnamed: 0']
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

# Final safety check for numeric types
non_numeric_cols = df.select_dtypes(exclude=['int64', 'float64', 'bool']).columns
if len(non_numeric_cols) > 0:
    # If any remain, attempt label-encoding just in case
    from sklearn.preprocessing import LabelEncoder
    for col in non_numeric_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    print(f"Applied label encoding for columns: {list(non_numeric_cols)}")

# Save final processed data
df.to_csv('final_data.csv', index=False)
print("✅ Feature engineering complete. Data saved to 'final_data.csv'")
