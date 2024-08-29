import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
import pickle

df = pd.read_csv('train_set_07292024_for_final_model_sm.csv')

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