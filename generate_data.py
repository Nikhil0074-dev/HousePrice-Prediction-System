"""
Generate realistic property dataset
Run: python generate_data.py
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)

# --- Property locations with price per sqft range (in Rs) ---
LOCATIONS = {
    "Gangapur Road":      (5500, 7500),
    "College Road":       (5000, 7000),
    "Canada Corner":      (4800, 6500),
    "Cidco":              (4000, 5500),
    "Mahatma Nagar":      (3800, 5000),
    "Dwarka":             (3500, 4800),
    "Panchavati":         (3500, 5000),
    "Nashik Road":        (3000, 4500),
    "Satpur":             (3200, 4200),
    "Ambad":              (3000, 4000),
    "Indira Nagar":       (2800, 3800),
    "Deolali":            (2800, 3800),
    "Trimbak Road":       (2500, 3500),
    "Pathardi Phata":     (2200, 3200),
    "Sinner Road":        (2000, 3000),
}

PROPERTY_TYPES = ["Apartment", "Villa", "Independent House"]
FURNISHED_STATUS = ["Unfurnished", "Semi-Furnished", "Furnished"]

# Property type multiplier on price
TYPE_MULTIPLIER = {
    "Apartment": 1.0,
    "Villa": 1.35,
    "Independent House": 1.15,
}

# Furnished multiplier
FURNISHED_MULTIPLIER = {
    "Unfurnished": 1.0,
    "Semi-Furnished": 1.07,
    "Furnished": 1.15,
}


def generate_property(idx):
    location = np.random.choice(list(LOCATIONS.keys()),
                                p=[0.10, 0.09, 0.08, 0.08, 0.07, 0.07, 0.07,
                                   0.07, 0.07, 0.07, 0.05, 0.05, 0.05, 0.04, 0.04])

    bedrooms = np.random.choice([1, 2, 3, 4], p=[0.15, 0.40, 0.35, 0.10])
    bathrooms = min(bedrooms, np.random.randint(1, bedrooms + 1))

    # Area based on bedrooms
    area_ranges = {1: (400, 700), 2: (650, 1100), 3: (950, 1600), 4: (1400, 2500)}
    area_sqft = np.random.randint(*area_ranges[bedrooms])

    property_type = np.random.choice(PROPERTY_TYPES, p=[0.60, 0.15, 0.25])
    age_years = np.random.randint(0, 21)
    floor = np.random.randint(0, 12) if property_type == "Apartment" else 0
    parking = np.random.choice([0, 1], p=[0.25, 0.75])
    furnished = np.random.choice(FURNISHED_STATUS, p=[0.35, 0.40, 0.25])

    # Price calculation
    min_rate, max_rate = LOCATIONS[location]
    rate_per_sqft = np.random.uniform(min_rate, max_rate)
    base_price = (area_sqft * rate_per_sqft) / 100000  # convert to lakhs

    # Apply multipliers
    price = base_price * TYPE_MULTIPLIER[property_type]
    price *= FURNISHED_MULTIPLIER[furnished]
    price *= (1 - age_years * 0.008)           # depreciation ~0.8% per year
    if parking == 1:
        price += np.random.uniform(1.5, 3.0)   # parking adds value
    if floor > 5:
        price *= 1.04                            # higher floor premium
    price += np.random.normal(0, price * 0.03)  # small noise
    price = max(5.0, round(price, 2))

    return {
        "id": idx,
        "location": location,
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "property_type": property_type,
        "age_years": age_years,
        "floor": floor,
        "parking": parking,
        "furnished": furnished,
        "price_lakhs": price,
    }


def main():
    records = [generate_property(i + 1) for i in range(600)]
    df = pd.DataFrame(records)

    os.makedirs("data", exist_ok=True)
    path = "data/houseprice_properties.csv"
    df.to_csv(path, index=False)
    print(f" Dataset saved: {path}  ({len(df)} rows)")
    print(df.describe())


if __name__ == "__main__":
    main()
