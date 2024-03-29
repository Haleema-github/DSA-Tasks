# Importing libraries
from flask import Flask, url_for, render_template, request
import pickle


import warnings
warnings.filterwarnings('ignore')

# Creating an instance of Flask
app=Flask(__name__)

# Loading the model
model=pickle.load(open('randomcv_grad_model.pkl','rb'))


# Creating homepage
@app.route('/')
def home():
    return render_template('index.html')

# Making Predictions
@app.route('/predict',methods=['POST'])
def predict():
    if request.method == "POST":

        Age = float(request.form['age'])

        Workclass = str(request.form['workclass'])

        Education = str(request.form['education'])

        Marital_status = str(request.form['marital-status'])

        Occupation = str(request.form['occupation'])

        Relationship = str(request.form['relationship'])

        Race = str(request.form['race'])

        Sex= str(request.form['sex'])

        Hours_per_week = float(request.form['hours-per-week'])

        Native_country = str(request.form['native-country'])

        # storing the data in 2-D array
        predict_list = [[Age,Workclass,Education,Marital_status,
                         Occupation,Relationship,Race,Sex,
                         Hours_per_week,Native_country]]
                            

# Predicting the results using the model loaded from a pickle file(randomcv_grad_model.pkl)
        output = model.predict(predict_list)

# Loading the templates for respective outputs (0 or 1)
        if output == 1:
            return render_template('high salary.html')
        else:
            return render_template('low salary.html')

    return render_template('index.html')


# Main driver function
if __name__ == '__main__':
    app.run(debug=True)
