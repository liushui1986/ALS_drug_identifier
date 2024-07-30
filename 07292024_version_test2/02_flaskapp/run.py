from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pandas as pd
import pickle
from rdkit import Chem
from mordred import Calculator, descriptors
import re
from sklearn.cluster import KMeans

app = Flask(__name__)

# Load the trained model
with open('trained_lgbm_model.pkl', 'rb') as f:
    kmeans, lgbm = pickle.load(f)

# Initialize the Mordred calculator
calc = Calculator(descriptors, ignore_3D=True)

# Extract processed features from the model file
# This assumes that 'trained_lgbm_model.pkl' includes the processed feature names
# If not, you'll need to provide these names manually based on the feature set used during model training
processed_features = pickle.load(open('processed_features.pkl', 'rb'))

def smiles_to_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Canonicalize SMILES string
        canon_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        mol = Chem.MolFromSmiles(canon_smiles)

        # Compute descriptors
        desc_matrix = calc.pandas([mol])
        desc = desc_matrix.iloc[0]

        # Clean column names
        desc.index = [re.sub(r'\W+', '_', col) for col in desc.index]
        desc = desc.to_frame().T
        
        # Ensure that only numerical columns are kept
        desc = desc.select_dtypes(include='number').astype('float32')

        # Filter descriptors to match model features
        desc = desc[processed_features]

        return desc
    except Exception as e:
        print(f"Error in smiles_to_descriptors: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    smiles = request.form.get('smiles', '').strip()

    # Convert SMILES to descriptors
    descriptors = smiles_to_descriptors(smiles)
    if descriptors is None:
        return render_template('index.html', prediction=False, error="Invalid SMILES string or unable to compute descriptors.")

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
        return render_template('index.html', prediction=True, result=result, probability=probability)
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return render_template('index.html', prediction=False, error="An error occurred while making the prediction.")

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)