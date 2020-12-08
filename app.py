# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 18:28:38 2020

@author: ram10
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if output==0:
        return render_template('index.html', prediction_text='No need to panic!!...\n You dont have diabetes.')
    elif output==1:
        return render_template('index.html', prediction_text='Consult a doctor!!...\n You have diabetes.')
    else : print('Invalid input')


if __name__ == "__main__":
    app.run(debug=True)