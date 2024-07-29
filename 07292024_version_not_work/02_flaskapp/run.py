from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('trained_lgbm_model.pkl', 'rb') as f:
    kmeans, lgbm = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    form_data = request.form.to_dict()
    form_data = {k: float(v) for k, v in form_data.items()}

    # Convert form data to DataFrame
    input_data = pd.DataFrame([form_data])

    # Make predictions
    cluster = kmeans.predict(input_data)[0]
    input_data['cluster'] = cluster
    prediction = lgbm.predict(input_data)[0]
    probability = lgbm.predict_proba(input_data)[0][1]

    # Return prediction result
    result = 'Good' if prediction == 1 else 'Bad'
    return render_template('index.html', prediction=True, result=result, probability=probability)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
