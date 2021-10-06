import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/", methods=['POST'])
def predict():
    age = request.form['age']
    sex = request.form['sex']
    bmi = request.form['bmi']
    children = request.form['children']
    smoker = request.form['smoker']
    region = request.form['region']
    
    model = pickle.load(open('model.pkl', 'rb'))
    data = [[age, sex, bmi, children, smoker, region]]
    
    df = pd.DataFrame(data, columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
    
    # perform feature encoding
    df['sex'] = np.where(df['sex'] == 'male', 0, 1)  # male - 0, female - 1
    df['smoker'] = np.where(df['smoker'] == 'yes', 0, 1)  # yes - 0, no - 1
    dict_region = {'southeast': 0,
               'southwest': 1,
               'northeast': 2,
               'northwest': 3}  # assigning value by using dict method
    df['region'] = df.region.map(dict_region)

    pred = model.predict(df)
    op = f"{np.round(pred[0],2)} $ is your future medical expenses!"
    
    return render_template('home.html', op=op, 
                           age=request.form['age'],
                           sex=request.form['sex'],
                           bmi=request.form['bmi'],
                           children=request.form['children'],
                           smoker=request.form['smoker'],
                           region=request.form['region'])
    
    
if __name__ == "__main__":
    app.run()