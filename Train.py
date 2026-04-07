import pandas as pd
import pickle
from sklearn.tree import DecisionTreeRegressor

# Load the coffee analysis data
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
df = pd.read_csv(url)

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

# Prepare the data - remove any rows with missing values
df_clean = df[['100g_USD', 'roast_cat', 'rating']].dropna()

X = df_clean[['100g_USD', 'roast_cat']].values
y = df_clean['rating'].values

# Train the Decision Tree Regressor model
model = DecisionTreeRegressor()
model.fit(X, y)

# Save the model as a pickle file
with open('model_2.pickle', 'wb') as f:
    pickle.dump(model, f)

print("Decision Tree model trained and saved as model_2.pickle")
