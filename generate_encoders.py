import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os

print("🔧 GENERATING ROBUST LABEL ENCODERS")
print("=" * 50)

if not os.path.exists("data.csv"):
    print("❌ Error: 'data.csv' not found.")
    exit(1)

df = pd.read_csv("data.csv")
print(f"✅ Dataset loaded. Shape: {df.shape}")

column_mappings = {
    'education': ['Education', 'education', 'Education Level', 'education_level', 'degree', 'Degree'],
    'job_role': ['Job Role', 'job_role', 'Job Title', 'job_title', 'Position', 'position', 'Role', 'role'],
    'location': ['Location', 'location', 'City', 'city', 'Region', 'region', 'Country', 'country'],
    'company_size': ['Company Size', 'company_size', 'companySize', 'company size'],
    'employment_type': ['Employment Type', 'employment_type', 'employmentType'],
    'experience_level': ['Experience Level', 'experience_level', 'experienceLevel'],
    'remote_ratio': ['Remote Ratio', 'remote_ratio'],
    'company_location': ['Company Location', 'company_location'],
    'employee_residence': ['Employee Residence', 'employee_residence', 'Residence'],
    'work_year': ['Work Year', 'work_year', 'Year']
}

actual_columns = {}
for standard_name, possible_names in column_mappings.items():
    for possible_name in possible_names:
        if possible_name in df.columns:
            actual_columns[standard_name] = possible_name
            break

# Use all object columns with fallback names if mapping missing
if not actual_columns:
    for col in df.select_dtypes(include=['object']).columns:
        actual_columns[col.lower().replace(' ', '_')] = col

if not actual_columns:
    print("❌ Error: No categorical columns found for encoding!")
    exit(1)

label_encoders = {}
for standard_name, actual_column in actual_columns.items():
    le = LabelEncoder()
    df[actual_column] = df[actual_column].fillna('Unknown')
    le.fit(df[actual_column])
    label_encoders[standard_name] = le
    print(f"Encoded: {standard_name} ({actual_column}) Values: {list(le.classes_)}")

joblib.dump(label_encoders, "preprocessor.pkl")
print("💾 Saved encoders to 'preprocessor.pkl'")

column_mapping_info = {
    'mappings': actual_columns,
    'reverse_mappings': {v: k for k, v in actual_columns.items()}
}
joblib.dump(column_mapping_info, "column_mappings.pkl")
print("💾 Saved column mappings to 'column_mappings.pkl'")
print("✅ ENCODER GENERATION COMPLETE!")
