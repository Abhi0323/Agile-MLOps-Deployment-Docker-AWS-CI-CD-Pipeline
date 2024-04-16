from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import input_data, Pred_Pipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Use the new names from the HTML form
        data = input_data(
            Type=request.form.get('Type'),
            Air_temperature=float((request.form.get('Air_temperature_K'))), 
            Process_temperature=float((request.form.get('Process_temperature_K'))),  
            Rotational_speed=(request.form.get('Rotational_speed_rpm')),  
            Torque=float((request.form.get('Torque_Nm'))),  
            Tool_wear=(request.form.get('Tool_wear_min'))  
        )
        pred_data = data.transfrom_data_as_dataframe()
        print(pred_data)
        print("Before Prediction")

        predict_pipeline = Pred_Pipeline()
        print("During Prediction")
        results = predict_pipeline.predict(pred_data)
        print("After Prediction")
         # Interpreting the model output
        if results[0] == 1:
            message = "There are high chances of machine failure soon. Immediate attention required."
        else:
            message = "There are no chances of machine failure. It is performing well for now."
        return render_template('home.html', results=message)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
