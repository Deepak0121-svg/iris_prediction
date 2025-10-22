# app.py

from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model
with open('KNeighbors_model.pkl', 'rb') as file:
    knn = pickle.load(file)

@app.route('/')
def home():
    result = ''
    return render_template('index.html', result=result)

@app.route('/predict', methods=['POST'])
def predict():
    #Id = int(request.form['Id'])
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    result = knn.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
