import pandas as pd
import numpy as np

def check_non_numeric_values(df, max_samples=5):
    print("\nChecking for non-numeric values in PCA data")
    
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"\nColumn '{col}':")
            print(f"Data type: {df[col].dtype}")
            print(f"Unique values sample: {df[col].unique()[:max_samples]}")
            
            non_numeric_samples = []
            for val in df[col].dropna().unique()[:max_samples]:
                try:
                    float(val)
                except (ValueError, TypeError):
                    non_numeric_samples.append(val)
            
            if non_numeric_samples:
                print(f"Non-numeric samples: {non_numeric_samples}")