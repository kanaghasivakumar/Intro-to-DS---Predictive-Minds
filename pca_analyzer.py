# pca_analyzer.py - FIXED VERSION
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from config import N_COMPONENTS

def clean_numeric_data(data):
    print("Ensuring all PCA data is numeric")
    data_clean = data.copy()
    
    problematic_columns = []
    
    for col in data_clean.columns:
        if data_clean[col].dtype == 'object':
            print(f"Converting column '{col}' to numeric")
            try:
                data_clean[col] = pd.to_numeric(data_clean[col], errors='coerce')
                
                if data_clean[col].isnull().mean() > 0.5:
                    problematic_columns.append(col)
                    print(f"Column '{col}' has too many non-numeric values - will be dropped")
                else:
                    median_val = data_clean[col].median()
                    data_clean[col] = data_clean[col].fillna(median_val)
                    print(f"Column '{col}' converted successfully")
                    
            except Exception as e:
                problematic_columns.append(col)
                print(f"Column '{col}' failed conversion: {e}")
    
    if problematic_columns:
        print(f"Dropping {len(problematic_columns)} problematic columns: {problematic_columns}")
        data_clean = data_clean.drop(columns=problematic_columns)
    
    non_numeric_cols = data_clean.select_dtypes(include=['object']).columns
    if len(non_numeric_cols) > 0:
        print(f"Still found non-numeric columns: {list(non_numeric_cols)} - dropping them")
        data_clean = data_clean.drop(columns=non_numeric_cols)
    
    print(f"Final PCA data shape: {data_clean.shape}")
    return data_clean

def perform_pca(data, n_components=N_COMPONENTS):
    print(" Performing PCA")
    
    data_clean = clean_numeric_data(data)
    
    if data_clean.shape[1] < 2:
        raise ValueError(f"Not enough numeric features for PCA. Only {data_clean.shape[1]} features remaining.")
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_clean)
    
    if n_components is None:
        pca_full = PCA()
        pca_full.fit(data_scaled)
        explained_variance = pca_full.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        n_components = np.argmax(cumulative_variance >= 0.95) + 1
        print(f"Selected {n_components} components explaining {cumulative_variance[n_components-1]:.2%} of variance")
    
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_scaled)
    
    pc_columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(principal_components, columns=pc_columns, index=data_clean.index)
    
    print("PCA completed")
    return pca, pca_df, scaler, data_scaled, data_clean.columns.tolist()

def analyze_pca_results(pca, feature_names, principal_df):
    print("\nPCA Results Analysis")
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print("Component Variance Explained:")
    for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
        print(f"PC{i+1}: {var:.3f} ({var:.2%}) - Cumulative: {cum_var:.2%}")
    
    components_df = pd.DataFrame(
        pca.components_,
        columns=feature_names,
        index=[f'PC{i+1}' for i in range(len(pca.components_))]
    )
    
    print("\nTop features for each principal component:")
    for i, component in enumerate(components_df.values):
        top_indices = np.argsort(np.abs(component))[-5:][::-1]
        print(f"\nPC{i+1}:")
        for idx in top_indices:
            feature = feature_names[idx]
            loading = component[idx]
            print(f"  {feature}: {loading:.3f}")
    
    return components_df, explained_variance, cumulative_variance