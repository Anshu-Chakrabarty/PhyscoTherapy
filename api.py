

from flask import Flask, render_template, request, jsonify
import pickle
import joblib

import numpy as np

app = Flask(__name__)


model = pickle.load(open(r"C:\Users\user\Desktop\Emotion-Detection using text\emotion_classifier_pipe_lr.pkl", 'rb'))

model1 = joblib.load(open(r"C:\Users\user\Desktop\Emotion-Detection using text\Mental_Health\stress.pkl", 'rb'))

model2 = joblib.load(open(r"C:\Users\user\Desktop\Emotion-Detection using text\Mental_Health\anxiety.pkl", 'rb'))

model3 = joblib.load(open(r"C:\Users\user\Desktop\Emotion-Detection using text\Mental_Health\depression.pkl", 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/emotions')
def get_emotion():
    return render_template('emotion.html')

@app.route('/stress')
def get_stress_page():
    return render_template('stress.html')

@app.route('/mentals')
def get_mental():
    return render_template('mental.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    int_value= [float(x) for x in request.form.values()]    
    final = [np.array(int_value)]
    prediction = model1.predict(final)[0]
    
    return render_template('stress.html', prediction=prediction)

@app.route('/predict', methods=['POST'])
def predict():
    
    input_text = request.form.values()
    
   
    prediction = model.predict(input_text)[0]

    
    return render_template('emotion.html', prediction=prediction)

@app.route('/anxiety')
def get_anxiety():
    return render_template('anxiety.html')

@app.route('/predicting', methods=['POST'])
def predict1():
    if request.method == 'POST':
        
        age = float(request.form['float_input'])
        gender = request.form['text_input']
        bmi = float(request.form['float_input'])
        who_bmi = request.form['text_input']
        gad_score = float(request.form['float_input'])

        input_data = np.array([[age, bmi, gad_score]])

        
        prediction_result = model2.predict(input_data)

        
        prediction_result = str(prediction_result[0])

        return render_template('anxiety.html', prediction=prediction_result)
    
@app.route('/depression')
def get_depression():
    return render_template('depression.html')

@app.route('/predicts', methods=['POST'])
def predict2():
     if request.method == 'POST':
        
        age = float(request.form['float_input'])
        gender = request.form['text_input']
        bmi = float(request.form['float_input'])
        who_bmi = request.form['text_input']
        phq_score = float(request.form['float_input'])

        input_data = np.array([[age, bmi, phq_score]])

        
        prediction_result = model3.predict(input_data)

        
        prediction_result = str(prediction_result[0])

        return render_template('depression.html', prediction=prediction_result)


if __name__ == '__main__':
    app.run(debug=True)
