import pandas as pd
import numpy as np
import re
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
import pickle

# Load the dataset
chemicals = pd.read_csv('train_set_original_data_07292024_for_final_model_sm.csv')
chemicals['canon_smiles'] = chemicals['canon_smiles'].astype(str)

# Canonicalize SMILES strings
def canonize(mol):
    return Chem.MolToSmiles(Chem.MolFromSmiles(mol), isomericSmiles=True, canonical=True)

chemicals['canon_smiles_new'] = chemicals['canon_smiles'].apply(canonize)

# Generate molecular objects
chemicals['mol'] = chemicals['canon_smiles_new'].apply(lambda x: Chem.MolFromSmiles(x))

# Calculate descriptors
calc = Calculator(descriptors, ignore_3D=True)
desc_matrix = calc.pandas(chemicals['mol'])

# Select numerical descriptors
desc_molecules = desc_matrix.select_dtypes(include=np.number).astype('float32')
desc_molecules = desc_molecules.loc[:, desc_molecules.var() > 0.0]

# Combine features and target variable
df = pd.concat([desc_molecules, chemicals[['good']]], axis=1)

# Define features and target variable
X = df.iloc[:, :-1] 
y = df.iloc[:, -1]

# Clean up feature names
def clean_column_names(df):
    df.columns = [re.sub(r'\W+', '_', col) for col in df.columns]
    return df

X = clean_column_names(X)

# Separate the majority and minority classes
X_majority = X[y == 0].copy()
X_minority = X[y == 1].copy()

# Perform clustering on the majority class
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
X_majority['cluster'] = kmeans.fit_predict(X_majority)

# Function to undersample within each cluster
def undersample_clusters(X_majority, target_ratio=0.1):
    undersampled_X = pd.DataFrame()
    undersampled_y = pd.Series(dtype='int')
    
    for cluster in X_majority['cluster'].unique():
        cluster_data = X_majority[X_majority['cluster'] == cluster]
        cluster_size = int(len(cluster_data) * target_ratio)
        undersampled_cluster = cluster_data.sample(cluster_size, random_state=42)
        
        undersampled_X = pd.concat([undersampled_X, undersampled_cluster])
        undersampled_y = pd.concat([undersampled_y, pd.Series([0] * cluster_size)])
    
    return undersampled_X.drop('cluster', axis=1), undersampled_y

# Undersample with the best ratio
X_majority_undersampled, y_majority_undersampled = undersample_clusters(X_majority, target_ratio=0.1)

# Combine the undersampled majority class with the minority class
X_combined = pd.concat([X_majority_undersampled, X_minority])
y_combined = pd.concat([y_majority_undersampled, y[y == 1]])

# Apply SMOTE to the combined data
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_combined, y_combined)

# Define and fit the LightGBM model
lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_resampled, y_resampled)

# Save the trained model
with open('trained_lgbm_model.pkl', 'wb') as f:
    pickle.dump((kmeans, lgbm), f)