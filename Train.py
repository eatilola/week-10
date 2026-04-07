# train.py

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pickle

# Dataset URL
DATA_URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"

# Function to map roast categories to numeric values
def roast_category(roast):
    mapping = {
        'Light': 0,
        'Medium-Light': 1,
        'Medium': 2,
        'Medium-Dark': 3,
        'Dark': 4
    }
    return mapping.get(roast, np.nan)  # unknown or missing → NaN

def main():
    # Load dataset
    df = pd.read_csv(DATA_URL)

    # Create numeric roast column
    df['roast_cat'] = df['roast'].apply(roast_category)

    # Select features and target
    df = df[['100g_USD', 'roast_cat', 'rating']]

    # Drop rows where target or price is missing (keep NaN in roast_cat if present)
    df = df.dropna(subset=['100g_USD', 'rating'])

    X = df[['100g_USD', 'roast_cat']]
    y = df['rating']

    # Train Decision Tree Regressor
    dtr = DecisionTreeRegressor(random_state=42)
    dtr.fit(X, y)

    # Save model
    with open("model_2.pickle", "wb") as f:
        pickle.dump(dtr, f)

    print("Model trained and saved as model_2.pickle")

if __name__ == "__main__":
    main()