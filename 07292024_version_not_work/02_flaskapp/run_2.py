from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pandas as pd
import pickle
from rdkit import Chem
from mordred import Calculator, descriptors
import re

app = Flask(__name__)

# Load the trained model
with open('trained_lgbm_model.pkl', 'rb') as f:
    kmeans, lgbm = pickle.load(f)

# Initialize the Mordred calculator
calc = Calculator(descriptors, ignore_3D=True)

def smiles_to_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Compute descriptors
        desc_matrix = calc.pandas([mol])
        desc = desc_matrix.iloc[0]

        # Clean column names
        desc.index = [re.sub(r'\W+', '_', col) for col in desc.index]
        desc = desc.to_frame().T
        
        # Ensure that only numerical columns are kept
        desc = desc.select_dtypes(include='number').astype('float32')

        return desc
    except Exception as e:
        print(f"Error in smiles_to_descriptors: {e}")
        return None

@app.route('/')
def index():
    return render_template('index_2.html')

@app.route('/predict', methods=['POST'])
def predict():
    smiles = request.form.get('smiles', '').strip()

    # Convert SMILES to descriptors
    descriptors = smiles_to_descriptors(smiles)
    if descriptors is None:
        return render_template('index_2.html', prediction=False, error="Invalid SMILES string or unable to compute descriptors.")

    # Ensure descriptors match the model's feature set
    required_columns = [col for col in descriptors.columns if col not in ['cluster']]  # Update this based on your feature set
    for col in required_columns:
        if col not in descriptors.columns:
            return render_template('index_2.html', prediction=False, error=f"Missing required descriptor: {col}")

    # Add a cluster column for prediction
    descriptors['cluster'] = kmeans.predict(descriptors)[0]
    
    # Convert descriptors to DataFrame
    input_data = pd.DataFrame(descriptors).T

    try:
        # Make predictions
        prediction = lgbm.predict(input_data)[0]
        probability = lgbm.predict_proba(input_data)[0][1]

        # Return prediction result
        result = 'Good' if prediction == 1 else 'Bad'
        return render_template('index_2.html', prediction=True, result=result, probability=probability)
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return render_template('index_2.html', prediction=False, error="An error occurred while making the prediction.")

       
if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
