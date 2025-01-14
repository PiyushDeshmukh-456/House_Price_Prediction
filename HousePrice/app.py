from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

data = pd.read_csv('Pune house data.csv')

data['site_location'] = data['site_location'].astype(str).fillna('Unknown')
locations = sorted(data['site_location'].unique())

@app.route('/')
def home():
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    location = features[-1]
    numeric_features = [float(x) for x in features[:-1]]
    
    in_data = pd.DataFrame([numeric_features + [location]], columns=['size', 'total_sqft', 'bath', 'balcony', 'site_location'])
    in_data = pd.get_dummies(in_data, columns=['site_location'])
    model_columns = model.feature_names_in_
    missing_cols = set(model_columns) - set(in_data.columns)
    for col in missing_cols:
        in_data[col] = 0
    in_data = in_data[model_columns]

    prediction = model.predict(in_data)
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text=f'₹{output} lakhs', locations=locations)


