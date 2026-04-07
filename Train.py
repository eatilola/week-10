import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Load the coffee analysis data
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
df = pd.read_csv(url)

# ===== EXERCISE 1 =====
# Prepare data for linear regression
df_clean_1 = df[['100g_USD', 'rating']].dropna()
X_1 = df_clean_1[['100g_USD']].values
y_1 = df_clean_1['rating'].values

# Train and save model 1
model_1 = LinearRegression()
model_1.fit(X_1, y_1)

with open('model_1.pickle', 'wb') as f:
    pickle.dump(model_1, f)

# ===== EXERCISE 2 =====
# Create a function to map roast values to numerical labels
def roast_category(roast_value):
    roast_map = {
        'Light': 1,
        'Medium-Light': 2,
        'Medium': 3,
        'Medium-Dark': 4,
        'Dark': 5
    }
    return roast_map.get(roast_value, roast_value)

# Apply the roast_category function to create roast_cat column
df['roast_cat'] = df['roast'].apply(roast_category)

# Prepare data for decision tree
df_clean_2 = df[['100g_USD', 'roast_cat', 'rating']].dropna()
X_2 = df_clean_2[['100g_USD', 'roast_cat']].values
y_2 = df_clean_2['rating'].values

# Train and save model 2
model_2 = DecisionTreeRegressor()
model_2.fit(X_2, y_2)

with open('model_2.pickle', 'wb') as f:
    pickle.dump(model_2, f)

print("Both models trained and saved!")
