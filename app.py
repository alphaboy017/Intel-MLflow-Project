# UI related entities
from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
import joblib
from datetime import datetime


app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def predict():
    if request.method == 'POST':
        try:
            # Read user inputs
            date_str = request.form['Date']
            milk_supply = float(request.form['Milk_Supply_Liters'])
            downtime = float(request.form['Downtime_Hours'])
            milk_500ml_inv = float(request.form['Milk_500ml_Inventory'])
            milk_1l_inv = float(request.form['Milk_1L_Inventory'])
            butter_inv = float(request.form['Butter_Inventory'])
            cheese_inv = float(request.form['Cheese_Inventory'])
            yogurt_inv = float(request.form['Yogurt_Inventory'])

            # Feature engineering (date features)
            date = pd.to_datetime(date_str)
            day_of_week = date.dayofweek
            month = date.month
            year = date.year

            # Prepare input for model (lags/rolling not available for single prediction)
            input_data = pd.DataFrame({
                'Date': [date],
                'Milk_Supply_Liters': [milk_supply],
                'Downtime_Hours': [downtime],
                'Milk_500ml_Inventory': [milk_500ml_inv],
                'Milk_1L_Inventory': [milk_1l_inv],
                'Butter_Inventory': [butter_inv],
                'Cheese_Inventory': [cheese_inv],
                'Yogurt_Inventory': [yogurt_inv],
                'day_of_week': [day_of_week],
                'month': [month],
                'year': [year]
            })

            # Load model and predict
            model = joblib.load('artifacts/model_trainer/model.joblib')
            # Drop target columns if present (for safety)
            target_columns = [
                'Milk_500ml_Demand', 'Milk_1L_Demand', 'Butter_Demand', 'Cheese_Demand', 'Yogurt_Demand'
            ]
            for col in target_columns:
                if col in input_data.columns:
                    input_data = input_data.drop(col, axis=1)
            prediction = model.predict(input_data)[0]
            prediction_dict = dict(zip(target_columns, prediction))

            return render_template('results.html', prediction=prediction_dict)
        except Exception as e:
            print('The Exception message is: ', e)
            return 'Something went wrong!'
    else:
        return render_template('index.html')


if __name__ == "__main__":
	# app.run(host="0.0.0.0", port = 8080, debug=True)
	app.run(host="0.0.0.0", port = 8080)