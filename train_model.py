"""
Train house price prediction models and save them.
Run: python train_model.py
"""
import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


CATEGORICAL_COLS = ["location", "property_type", "furnished"]
FEATURE_COLS = [
    "area_sqft", "bedrooms", "bathrooms",
    "location", "property_type", "age_years",
    "floor", "parking", "furnished"
]
TARGET_COL = "price_lakhs"


def load_and_preprocess(csv_path="data/houseprice_properties.csv"):
    df = pd.read_csv(csv_path)
    encoders = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y, encoders, df


def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    mse  = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, preds)
    print(f"\n {name}")
    print(f"   MAE  : {mae:.2f} lakhs")
    print(f"   RMSE : {rmse:.2f} lakhs")
    print(f"   R²   : {r2:.4f}")
    return {"name": name, "mae": round(mae, 2), "mse": round(mse, 2),
            "rmse": round(rmse, 2), "r2": round(r2, 4)}


def train_all():
    print(" Loading data …")
    X, y, encoders, _ = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    os.makedirs("models", exist_ok=True)

    models_info = []

    # --- Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    info = evaluate(lr, X_test, y_test, "Linear Regression")
    joblib.dump(lr, "models/linear_regression.pkl")
    info["file"] = "models/linear_regression.pkl"
    models_info.append(info)

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=200, max_depth=12,
                               min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    info = evaluate(rf, X_test, y_test, "Random Forest")
    joblib.dump(rf, "models/random_forest.pkl")
    info["file"] = "models/random_forest.pkl"
    models_info.append(info)

    # --- Gradient Boosting ---
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                   max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    info = evaluate(gb, X_test, y_test, "Gradient Boosting")
    joblib.dump(gb, "models/gradient_boosting.pkl")
    info["file"] = "models/gradient_boosting.pkl"
    models_info.append(info)

    # --- XGBoost (if available) ---
    if XGBOOST_AVAILABLE:
        xgb = XGBRegressor(n_estimators=200, learning_rate=0.05,
                           max_depth=6, random_state=42,
                           verbosity=0, eval_metric="rmse")
        xgb.fit(X_train, y_train)
        info = evaluate(xgb, X_test, y_test, "XGBoost")
        joblib.dump(xgb, "models/xgboost.pkl")
        info["file"] = "models/xgboost.pkl"
        models_info.append(info)

    # Save encoders
    joblib.dump(encoders, "models/encoders.pkl")
    joblib.dump(FEATURE_COLS, "models/feature_cols.pkl")

    # Save best model reference (highest R²)
    best = max(models_info, key=lambda m: m["r2"])
    with open("models/best_model.json", "w") as f:
        json.dump(best, f, indent=2)

    # Save all metrics
    with open("models/all_metrics.json", "w") as f:
        json.dump(models_info, f, indent=2)

    print(f"\n Best model: {best['name']}  R²={best['r2']}")
    print(" All models saved to models/")
    return best


if __name__ == "__main__":
    train_all()
