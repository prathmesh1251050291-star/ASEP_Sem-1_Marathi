import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Sample dataset creation (replace with your actual data)
# Features: Rainfall(mm), Humidity(%), Windspeed(km/h), Temperature(°C)
np.random.seed(42)
data = {
    'Year': list(range(2010, 2024)),
    'Rainfall': np.random.uniform(800, 1500, 14),
    'Humidity': np.random.uniform(60, 90, 14),
    'Windspeed': np.random.uniform(5, 20, 14),
    'Temperature': np.random.uniform(22, 32, 14),
    'Yield': np.random.uniform(2500, 4500, 14)  # kg/ha
}

df = pd.DataFrame(data)

# Prepare features and target
X = df[['Rainfall', 'Humidity', 'Windspeed', 'Temperature']]
y = df['Yield']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Save the model
with open('ASEP_Crop.pkl', 'wb') as f:
    pickle.dump(model, f)

# Predict for a new year
new_data = pd.DataFrame({
    'Rainfall': [1200],
    'Humidity': [75],
    'Windspeed': [12],
    'Temperature': [27]
})

predicted_yield = model.predict(new_data)[0]
area_eastern_pune = 15000  # hectares (adjust based on actual data)
total_production = predicted_yield * area_eastern_pune

# Determine surplus or deficit (assuming demand is 50,000,000 kg)
demand = 50000000  # kg for eastern Pune region
print(f"\nPredicted Yield: {predicted_yield:.2f} kg/ha")
print(f"Total Production: {total_production:.2f} kg")

if total_production > demand:
    print(f"Status: SURPLUS ({total_production - demand:.2f} kg)")
else:
    print(f"Status: DEFICIT ({demand - total_production:.2f} kg)")

# Feature importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(importance)
