from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
model=joblib.load('titanic_model.pkl')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    sex=request.form['sex']
    if sex=='male' or sex=='Male':
        sex=0
    elif sex=='female' or sex=='Female':
        sex=1

    age = int(request.form['age'])
    pclass = int(request.form['pclass'])
    fare = float(request.form['fare'])
    sibsp = int(request.form['sibsp'])
    parch = int(request.form['parch'])

    final=[[sex,age,pclass,fare,sibsp,parch]]
    pred=model.predict(final)
    if pred[0]==0:
        result="Passenger did not Survive 😔"
    else :
        result="Passenger Survived 😊"
    return render_template(
        'index.html',
        prediction_text=result
    )

if __name__ == '__main__':

    port = int(os.environ.get('PORT', 5000))

    app.run(
        host='0.0.0.0',
        port=port,
        debug=True
    )
