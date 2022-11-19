import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request


app = Flask(__name__)
model = pickle.load(open("WE4.pkl",'rb'))

@app.route("/",methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/Predict", methods=['POST'])
def predict():
    flightno = int(request.form['NO'])
    month = int(request.form['month'])
    dayofmonth = int(request.form['dayofmonth'])
    dayofweek = int(request.form['dayofweek'])
    origin = int(request.form['origin'])
    destination = int(request.form['dest'])
    crs = int(request.form['crr'])
    depature_time = int(request.form['deptime'])

    X = [[flightno,month, dayofmonth, dayofweek, origin, destination, crs , depature_time]]
    prediction = model.predict(X)

    return render_template('results.html', prediction_text=int(prediction))

if __name__=="__main__":
    app.run(debug=True)