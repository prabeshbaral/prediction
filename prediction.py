from dataprocessing import calculate_price
from flask import Flask, render_template, request
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    longitude = float(request.form['longitude'])
    latitude = float(request.form['latitude'])
    housing_median_age = float(request.form['housing_median_age'])
    total_rooms1 = float(request.form['total_rooms'])
    total_bedrooms1 = float(request.form['total_bedrooms'])
    population1 = float(request.form['population'])
    households1 = float(request.form['households'])
    median_income = float(request.form['median_income'])
    proximity = request.form['proximity']

    total_rooms = np.log(total_rooms1 + 1)
    total_bedrooms = np.log(total_bedrooms1 + 1)
    population = np.log(population1 + 1)
    households = np.log(households1 + 1)
    
    bedroom_ratio = total_bedrooms / total_rooms if total_rooms != 0 else 0
    household_room = total_rooms / households if households != 0 else 0


    # One-hot encoding for proximity variables
    proximity_map = {
        '<1H OCEAN': [1, 0, 0, 0, 0],
        'INLAND': [0, 1, 0, 0, 0],
        'ISLAND': [0, 0, 1, 0, 0],
        'NEAR BAY': [0, 0, 0, 1, 0],
        'NEAR OCEAN': [0, 0, 0, 0, 1]
    }
    proximity_features = proximity_map[proximity]

    # Combine all features into a single input array
    feature = [[longitude, latitude, housing_median_age, total_rooms,
                          total_bedrooms, population, households, median_income] + proximity_features+[bedroom_ratio,household_room]]
    
    
    predicted_value = calculate_price(feature)
    #predicted_value = calculate_price(for_test)
    print("Predicted Value:", predicted_value)

    # Render prediction
    return render_template('index.html', prediction=f'Median House Value: Rs{predicted_value}')

if __name__ == '__main__':
    app.run(debug=True)