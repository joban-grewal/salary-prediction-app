import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Optional extra models
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Load data
df = pd.read_csv('final_data.csv')
target_columns = [col for col in df.columns if 'salary' in col.lower()]
if not target_columns:
    raise Exception("No salary column found in the data!")
target_col = target_columns[0]

X = df.drop(target_col, axis=1)
y = df[target_col]

# Ensure all X columns are numeric (float)
X = X.apply(pd.to_numeric, errors='coerce')
X = X.astype(float)
# Drop rows with any NaN
is_good = (~X.isna().any(axis=1))
X = X[is_good]
y = y[is_good]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100),
}
if XGBOOST_AVAILABLE:
    models['XGBoost'] = XGBRegressor(random_state=42, n_estimators=100)
if LIGHTGBM_AVAILABLE:
    models['LightGBM'] = LGBMRegressor(random_state=42, n_estimators=100, verbose=-1)

results = {}
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'model': mdl, 'R2': r2}

best_name = max(results, key=lambda x: results[x]['R2'])
best_model = results[best_name]['model']

# Save model and info
joblib.dump(best_model, 'salary_model.pkl')
model_info = {
    'model_name': best_name,
    'feature_names': list(X.columns),
    'target_name': target_col
}
joblib.dump(model_info, 'model_info.pkl')
print(f"✅ Saved best model {best_name} and info. Top R2={results[best_name]['R2']:.4f}")
