import pandas as pd
from config import DATA_PATH

def load_data(file_path=DATA_PATH):
    """Load and initially explore the dataset"""
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        return None

def explore_data(df):
    """Perform initial data exploration"""
    print("\n--- Data Exploration ---")
    print("Data types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nBasic statistics:\n", df.describe())
    
    return df