"""
ML Service – loads trained models and runs predictions.
"""
import joblib
import json
import os
import numpy as np
import pandas as pd

MODELS_DIR = "models"

_encoders      = None
_feature_cols  = None
_active_model  = None
_active_name   = ""


def _load_assets():
    global _encoders, _feature_cols
    if _encoders is None:
        _encoders     = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
        _feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))


def load_model(file_path, model_name):
    global _active_model, _active_name
    _load_assets()
    _active_model = joblib.load(file_path)
    _active_name  = model_name


def get_active_model_name():
    return _active_name


def predict(data: dict) -> float:
    """
    data keys: location, area_sqft, bedrooms, bathrooms,
               property_type, age_years, floor, parking, furnished
    Returns predicted price in lakhs (float).
    """
    _load_assets()
    if _active_model is None:
        raise RuntimeError("No ML model loaded. Run train_model.py first.")

    row = {}
    for col in _feature_cols:
        val = data.get(col)
        if col in _encoders:
            enc = _encoders[col]
            if val not in enc.classes_:
                val = enc.classes_[0]         # fallback to first class
            val = enc.transform([val])[0]
        row[col] = val

    X = pd.DataFrame([row], columns=_feature_cols)
    price = float(_active_model.predict(X)[0])
    return round(max(price, 5.0), 2)


def get_all_metrics():
    path = os.path.join(MODELS_DIR, "all_metrics.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def get_best_model_info():
    path = os.path.join(MODELS_DIR, "best_model.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


LOCATIONS = [
    "Gangapur Road", "College Road", "Canada Corner",
    "Cidco", "Mahatma Nagar", "Dwarka", "Panchavati",
    "Nashik Road", "Satpur", "Ambad",
    "Indira Nagar", "Deolali", "Trimbak Road",
    "Pathardi Phata", "Sinner Road",
]

PROPERTY_TYPES  = ["Apartment", "Villa", "Independent House"]
FURNISHED_OPTS  = ["Unfurnished", "Semi-Furnished", "Furnished"]
