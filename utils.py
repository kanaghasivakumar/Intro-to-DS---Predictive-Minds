import pandas as pd
from config import OUTPUT_PATH, PCA_COMPONENTS_PATH

def save_results(cleaned_df, principal_df):
    """Save processed data and PCA results"""
    print("💾 Saving results...")
    
    cleaned_df.to_csv(OUTPUT_PATH, index=False)
    principal_df.to_csv(PCA_COMPONENTS_PATH)
    
    print(f"✅ Processed data saved to: {OUTPUT_PATH}")
    print(f"✅ PCA components saved to: {PCA_COMPONENTS_PATH}")

def print_summary(original_df, cleaned_df):
    """Print summary of the processing"""
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Original dataset shape: {original_df.shape}")
    print(f"Processed dataset shape: {cleaned_df.shape}")
    print(f"Original columns: {len(original_df.columns)}")
    print(f"Processed columns: {len(cleaned_df.columns)}")
    print("="*50)