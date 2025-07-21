import pandas as pd
import joblib
import os

print("🔍 DEBUGGING ENCODER ISSUE")
print("=" * 50)

# 1. Check if files exist
print("📁 File Check:")
files_to_check = ["data.csv", "preprocessor.pkl", "salary_model.pkl"]
for file in files_to_check:
    exists = "✅" if os.path.exists(file) else "❌"
    print(f"  {exists} {file}")

print("\n" + "=" * 50)

# 2. Load and examine the original dataset
if os.path.exists("data.csv"):
    print("📊 Original Dataset Analysis:")
    df = pd.read_csv("data.csv")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"Categorical columns: {list(categorical_cols)}")
    
    # Show unique values for each categorical column
    for col in categorical_cols:
        unique_vals = df[col].unique()
        print(f"\n{col}:")
        print(f"  Unique values ({len(unique_vals)}): {list(unique_vals)}")
else:
    print("❌ data.csv not found!")

print("\n" + "=" * 50)

# 3. Load and examine the preprocessor
if os.path.exists("preprocessor.pkl"):
    print("🔧 Preprocessor Analysis:")
    try:
        encoders = joblib.load("preprocessor.pkl")
        print(f"Type: {type(encoders)}")
        
        if isinstance(encoders, dict):
            print(f"Number of encoders: {len(encoders)}")
            print(f"Encoder keys: {list(encoders.keys())}")
            
            for key, encoder in encoders.items():
                print(f"\n{key}:")
                print(f"  Type: {type(encoder)}")
                if hasattr(encoder, 'classes_'):
                    print(f"  Classes: {list(encoder.classes_)}")
        else:
            print(f"❌ Preprocessor is not a dictionary! It's: {type(encoders)}")
            
    except Exception as e:
        print(f"❌ Error loading preprocessor: {e}")
else:
    print("❌ preprocessor.pkl not found!")

print("\n" + "=" * 50)
print("🔧 SOLUTION RECOMMENDATIONS:")
print("=" * 50)

# Provide solutions based on what we found
if not os.path.exists("data.csv"):
    print("1. ❌ Create or place your data.csv file in the current directory")

if not os.path.exists("preprocessor.pkl"):
    print("2. ❌ Run: python generate_encoders.py")
elif os.path.exists("data.csv"):
    # Check column name mismatches
    df = pd.read_csv("data.csv")
    expected_cols = ['Education', 'Job Role', 'Location']
    actual_cols = list(df.columns)
    
    print("3. 🔍 Column Name Check:")
    for col in expected_cols:
        if col in actual_cols:
            print(f"   ✅ {col} - Found")
        else:
            # Look for similar columns
            similar = [c for c in actual_cols if col.lower() in c.lower() or c.lower() in col.lower()]
            if similar:
                print(f"   ❌ {col} - Not found, but similar: {similar}")
            else:
                print(f"   ❌ {col} - Not found")
    
    print(f"   📋 All columns in dataset: {actual_cols}")

print("\n4. 💡 Quick Fix Commands:")
print("   python generate_encoders.py  # Regenerate encoders")
print("   python debug_encoders.py     # Run this script again to verify")