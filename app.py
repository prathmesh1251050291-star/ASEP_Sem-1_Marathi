from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('rice_yield_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        rainfall = float(request.form['rainfall'])
        humidity = float(request.form['humidity'])
        windspeed = float(request.form['windspeed'])
        temperature = float(request.form['temperature'])
        
        # Create dataframe for prediction
        input_data = pd.DataFrame({
            'Rainfall': [rainfall],
            'Humidity': [humidity],
            'Windspeed': [windspeed],
            'Temperature': [temperature]
        })
        
        # Make prediction
        predicted_yield = model.predict(input_data)[0]
        
        # Calculate total production (assuming 15000 hectares in Eastern Pune)
        area = 15000
        total_production = predicted_yield * area
        
        # Determine surplus or deficit
        demand = 50000000  # kg
        if total_production > demand:
            status = "SURPLUS"
            difference = total_production - demand
        else:
            status = "DEFICIT"
            difference = demand - total_production
        
        return render_template('index.html', 
                             prediction=f'{predicted_yield:.2f}',
                             total_production=f'{total_production:,.0f}',
                             status=status,
                             difference=f'{difference:,.0f}',
                             rainfall=rainfall,
                             humidity=humidity,
                             windspeed=windspeed,
                             temperature=temperature)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)