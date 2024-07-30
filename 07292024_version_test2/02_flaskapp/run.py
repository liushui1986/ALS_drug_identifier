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

# Load the selected features
selected_features = ['nAcid', 'nBase', 'nAromAtom', 'nAromBond', 'nAtom', 'nHeavyAtom', 'nSpiro', 'nBridgehead', 'nHetero', 'nH', 'nB', 'nC', 'nN', 'nO', 'nS', 'nP', 'nF', 'nCl', 'nBr', 'nI', 'nX', 'ATS0dv', 'ATS1dv', 'ATS2dv', 'ATS3dv', 'ATS4dv', 'ATS5dv', 'ATS6dv', 'ATS7dv', 'ATS8dv', 'ATS0d', 'ATS1d', 'ATS2d', 'ATS3d', 'ATS4d', 'ATS5d', 'ATS6d', 'ATS7d', 'ATS8d', 'ATS0Z', 'ATS1Z', 'ATS2Z', 'ATS3Z', 'ATS4Z', 'ATS5Z', 'ATS6Z', 'ATS7Z', 'ATS8Z', 'ATS0m', 'ATS1m', 'ATS2m', 'ATS3m', 'ATS4m', 'ATS5m', 'ATS6m', 'ATS7m', 'ATS8m', 'ATS0v', 'ATS1v', 'ATS2v', 'ATS3v', 'ATS4v', 'ATS5v', 'ATS6v', 'ATS7v', 'ATS8v', 'ATS0pe', 'ATS1pe', 'ATS2pe', 'ATS3pe', 'ATS4pe', 'ATS5pe', 'ATS6pe', 'ATS7pe', 'ATS8pe', 'ATS0are', 'ATS1are', 'ATS2are', 'ATS3are', 'ATS4are', 'ATS5are', 'ATS6are', 'ATS7are', 'ATS8are', 'ATS0p', 'ATS1p', 'ATS2p', 'ATS3p', 'ATS4p', 'ATS5p', 'ATS6p', 'ATS7p', 'ATS8p', 'ATS0i', 'ATS1i', 'ATS2i', 'ATS3i', 'ATS4i', 'ATS5i', 'ATS6i', 'ATS7i', 'ATS8i', 'AATS0dv', 'AATS0d', 'AATS0Z', 'AATS0m', 'AATS0v', 'AATS0pe', 'AATS0are', 'AATS0p', 'AATS0i', 'ATSC0dv', 'ATSC1dv', 'ATSC2dv', 'ATSC3dv', 'ATSC4dv', 'ATSC5dv', 'ATSC6dv', 'ATSC7dv', 'ATSC8dv', 'ATSC0d', 'ATSC1d', 'ATSC2d', 'ATSC3d', 'ATSC4d', 'ATSC5d', 'ATSC6d', 'ATSC7d', 'ATSC8d', 'ATSC0Z', 'ATSC1Z', 'ATSC2Z', 'ATSC3Z', 'ATSC4Z', 'ATSC5Z', 'ATSC6Z', 'ATSC7Z', 'ATSC8Z', 'ATSC0m', 'ATSC1m', 'ATSC2m', 'ATSC3m', 'ATSC4m', 'ATSC5m', 'ATSC6m', 'ATSC7m', 'ATSC8m', 'ATSC0v', 'ATSC1v', 'ATSC2v', 'ATSC3v', 'ATSC4v', 'ATSC5v', 'ATSC6v', 'ATSC7v', 'ATSC8v', 'ATSC0pe', 'ATSC1pe', 'ATSC2pe', 'ATSC3pe', 'ATSC4pe', 'ATSC5pe', 'ATSC6pe', 'ATSC7pe', 'ATSC8pe', 'ATSC0are', 'ATSC1are', 'ATSC2are', 'ATSC3are', 'ATSC4are', 'ATSC5are', 'ATSC6are', 'ATSC7are', 'ATSC8are', 'ATSC0p', 'ATSC1p', 'ATSC2p', 'ATSC3p', 'ATSC4p', 'ATSC5p', 'ATSC6p', 'ATSC7p', 'ATSC8p', 'ATSC0i', 'ATSC1i', 'ATSC2i', 'ATSC3i', 'ATSC4i', 'ATSC5i', 'ATSC6i', 'ATSC7i', 'ATSC8i', 'AATSC0dv', 'AATSC0d', 'AATSC0Z', 'AATSC0m', 'AATSC0v', 'AATSC0pe', 'AATSC0are', 'AATSC0p', 'AATSC0i', 'BalabanJ', 'BertzCT', 'nBonds', 'nBondsO', 'nBondsS', 'nBondsD', 'nBondsT', 'nBondsA', 'nBondsM', 'nBondsKS', 'nBondsKD', 'C1SP1', 'C2SP1', 'C1SP2', 'C2SP2', 'C3SP2', 'C1SP3', 'C2SP3', 'C3SP3', 'C4SP3', 'FCSP3', 'Xch_3d', 'Xch_4d', 'Xch_5d', 'Xch_6d', 'Xch_7d', 'Xch_3dv', 'Xch_4dv', 'Xch_5dv', 'Xch_6dv', 'Xch_7dv', 'Xc_3d', 'Xc_4d', 'Xc_5d', 'Xc_6d', 'Xc_3dv', 'Xc_4dv', 'Xc_5dv', 'Xc_6dv', 'Xpc_4d', 'Xpc_5d', 'Xpc_6d', 'Xpc_4dv', 'Xpc_5dv', 'Xpc_6dv', 'Xp_1d', 'Xp_2d', 'Xp_3d', 'Xp_4d', 'Xp_5d', 'Xp_6d', 'Xp_7d', 'Xp_1dv', 'Xp_2dv', 'Xp_3dv', 'Xp_4dv', 'Xp_5dv', 'Xp_6dv', 'Xp_7dv', 'SZ', 'Sm', 'Sv', 'Spe', 'Sare', 'Sp', 'Si', 'MZ', 'Mm', 'Mv', 'Mpe', 'Mare', 'Mp', 'Mi', 'NsssB', 'NsCH3', 'NdCH2', 'NssCH2', 'NtCH', 'NdsCH', 'NaaCH', 'NsssCH', 'NddC', 'NtsC', 'NdssC', 'NaasC', 'NaaaC', 'NssssC', 'NsNH2', 'NdNH', 'NssNH', 'NaaNH', 'NtN', 'NdsN', 'NaaN', 'NsssN', 'NddsN', 'NaasN', 'NssssN', 'NsOH', 'NdO', 'NssO', 'NaaO', 'NsF', 'NdsssP', 'NsSH', 'NdS', 'NssS', 'NaaS', 'NdssS', 'NddssS', 'NsCl', 'NsBr', 'NsI', 'SsssB', 'SsCH3', 'SdCH2', 'SssCH2', 'StCH', 'SdsCH', 'SaaCH', 'SsssCH', 'SddC', 'StsC', 'SdssC', 'SaasC', 'SaaaC', 'SssssC', 'SsNH2', 'SdNH', 'SssNH', 'SaaNH', 'StN', 'SdsN', 'SaaN', 'SsssN', 'SddsN', 'SaasN', 'SssssN', 'SsOH', 'SdO', 'SssO', 'SaaO', 'SsF', 'SdsssP', 'SsSH', 'SdS', 'SssS', 'SaaS', 'SdssS', 'SddssS', 'SsCl', 'SsBr', 'SsI', 'ECIndex', 'fragCpx', 'fMF', 'nHBAcc', 'nHBDon', 'IC0', 'IC1', 'IC2', 'IC3', 'IC4', 'IC5', 'TIC0', 'TIC1', 'TIC2', 'TIC3', 'TIC4', 'TIC5', 'SIC0', 'SIC1', 'SIC2', 'SIC3', 'SIC4', 'SIC5', 'CIC0', 'CIC1', 'CIC2', 'CIC3', 'CIC4', 'CIC5', 'MIC0', 'MIC1', 'MIC2', 'MIC3', 'MIC4', 'MIC5', 'ZMIC0', 'ZMIC1', 'ZMIC2', 'ZMIC3', 'ZMIC4', 'ZMIC5', 'FilterItLogS', 'VMcGowan', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA10', 'SlogP_VSA11', 'EState_VSA1', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'EState_VSA10', 'VSA_EState1', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'MPC2', 'MPC3', 'MPC4', 'MPC5', 'MPC6', 'MPC7', 'MPC8', 'MPC9', 'MPC10', 'TMPC10', 'piPC1', 'piPC2', 'piPC3', 'piPC4', 'piPC5', 'piPC6', 'piPC7', 'piPC8', 'piPC9', 'piPC10', 'TpiPC10', 'apol', 'bpol', 'nRing', 'n3Ring', 'n4Ring', 'n5Ring', 'n6Ring', 'n7Ring', 'n8Ring', 'n9Ring', 'n10Ring', 'n12Ring', 'nG12Ring', 'nHRing', 'n3HRing', 'n4HRing', 'n5HRing', 'n6HRing', 'n7HRing', 'n8HRing', 'n9HRing', 'n12HRing', 'nG12HRing', 'naRing', 'n5aRing', 'n6aRing', 'n7aRing', 'naHRing', 'n5aHRing', 'n6aHRing', 'nARing', 'n3ARing', 'n4ARing', 'n5ARing', 'n6ARing', 'n7ARing', 'n8ARing', 'n9ARing', 'n10ARing', 'n12ARing', 'nG12ARing', 'nAHRing', 'n3AHRing', 'n4AHRing', 'n5AHRing', 'n6AHRing', 'n7AHRing', 'n8AHRing', 'n9AHRing', 'n12AHRing', 'nG12AHRing', 'nFRing', 'n6FRing', 'n7FRing', 'n8FRing', 'n9FRing', 'n10FRing', 'n11FRing', 'n12FRing', 'nG12FRing', 'nFHRing', 'n6FHRing', 'n7FHRing', 'n8FHRing', 'n9FHRing', 'n10FHRing', 'n11FHRing', 'n12FHRing', 'nG12FHRing', 'nFaRing', 'n8FaRing', 'n9FaRing', 'n10FaRing', 'n12FaRing', 'nG12FaRing', 'nFaHRing', 'n8FaHRing', 'n9FaHRing', 'n10FaHRing', 'n12FaHRing', 'nG12FaHRing', 'nFARing', 'n6FARing', 'n7FARing', 'n8FARing', 'n9FARing', 'n10FARing', 'n11FARing', 'n12FARing', 'nG12FARing', 'nFAHRing', 'n6FAHRing', 'n7FAHRing', 'n8FAHRing', 'n9FAHRing', 'n10FAHRing', 'n11FAHRing', 'n12FAHRing', 'nG12FAHRing', 'nRot', 'SLogP', 'SMR', 'TopoPSA_NO_', 'TopoPSA', 'GGI1', 'GGI2', 'GGI3', 'GGI4', 'GGI5', 'GGI6', 'GGI7', 'GGI8', 'GGI9', 'GGI10', 'JGI1', 'JGI2', 'JGI3', 'JGI4', 'JGI5', 'JGI6', 'JGI7', 'JGI8', 'JGI9', 'JGI10', 'JGT10', 'Diameter', 'Radius', 'TopoShapeIndex', 'PetitjeanIndex', 'MWC01', 'MWC02', 'MWC03', 'MWC04', 'MWC05', 'MWC06', 'MWC07', 'MWC08', 'MWC09', 'MWC10', 'TMWC10', 'SRW02', 'SRW03', 'SRW04', 'SRW05', 'SRW06', 'SRW07', 'SRW08', 'SRW09', 'SRW10', 'TSRW10', 'MW', 'AMW', 'WPath', 'WPol', 'Zagreb1', 'Zagreb2', 'mZagreb2']

# Initialize the Mordred calculator
calc = Calculator(descriptors, ignore_3D=True)

def smiles_to_descriptors(smiles_list):
    try:
        # Canonicalize SMILES strings
        canon_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True, canonical=True) for smiles in smiles_list]
        mols = [Chem.MolFromSmiles(smiles) for smiles in canon_smiles]

        # Calculate descriptors
        desc_matrix = calc.pandas(mols)

        # Select numerical descriptors and ensure they match the selected features
        desc_molecules = desc_matrix.select_dtypes(include=np.number).astype('float32')

        # Add missing features as zero and remove extra features
        for feature in selected_features:
            if feature not in desc_molecules.columns:
                desc_molecules[feature] = 0
        
        # Keep only the selected features
        desc_molecules = desc_molecules[selected_features]

        return desc_molecules
    except Exception as e:
        print(f"Error in smiles_to_descriptors: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    smiles = request.form.get('smiles', '').strip().split()

    # Convert SMILES to descriptors
    descriptors = smiles_to_descriptors(smiles)
    if descriptors is None or descriptors.empty:
        return render_template('index.html', prediction=False, error="Invalid SMILES string or unable to compute descriptors.")

    try:
        # Ensure the number of features matches the model's expected input
        if descriptors.shape[1] != 626:  # Expecting exactly 626 features
            return render_template('index.html', prediction=False, error="Feature number mismatch.")

        # Make predictions
        prediction = lgbm.predict(descriptors)[0]
        probability = lgbm.predict_proba(descriptors)[0][1]

        # Return prediction result
        result = 'Good' if prediction == 1 else 'Bad'
        return render_template('index.html', prediction=True, result=result, probability=probability)
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return render_template('index.html', prediction=False, error="An error occurred while making the prediction.")

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)