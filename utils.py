import pandas as pd
from config import OUTPUT_PATH, PCA_COMPONENTS_PATH, OUTPUT_PATH2, OUTPUT_PATH3

def save_results(cleaned_df, principal_df):
    print(" Saving results")
    
    # cleaned_df.to_csv(OUTPUT_PATH, index=False)
    # principal_df.to_csv(PCA_COMPONENTS_PATH)
    
    cleaned_df.to_csv(OUTPUT_PATH2, index=False)
    principal_df.to_csv(OUTPUT_PATH3)
    
    # print(f" Processed data saved to: {OUTPUT_PATH}")
    # print(f" PCA components saved to: {PCA_COMPONENTS_PATH}")

    print(f" Cleaned data saved to: {OUTPUT_PATH2}")
    print(f" Cleaned and missing values handled and saved to: {OUTPUT_PATH3}")

def print_summary(original_df, cleaned_df):
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Original dataset shape: {original_df.shape}")
    print(f"Processed dataset shape: {cleaned_df.shape}")
    print(f"Original columns: {len(original_df.columns)}")
    print(f"Processed columns: {len(cleaned_df.columns)}")
    print("="*50)