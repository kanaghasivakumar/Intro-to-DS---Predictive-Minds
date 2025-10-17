# main.py - CORRECTED VERSION
from data_loader import load_data, explore_data
from data_cleaner import clean_data, handle_missing_values
from feature_engineer import engineer_features, select_pca_features
from pca_analyzer import perform_pca, analyze_pca_results
from utils import save_results, print_summary
from debug_utils import check_non_numeric_values

def main():
    """Main execution function"""
    print("🚀 Starting LA Airbnb Data Analysis Pipeline...\n")
    
    # Step 1: Load and explore data
    df = load_data()
    if df is None:
        return
    
    explore_data(df)
    
    # Step 2: Data cleaning
    df_clean = clean_data(df)
    print(f"After cleaning - Shape: {df_clean.shape}")
    
    df_filled = handle_missing_values(df_clean)
    print(f"After missing values - Shape: {df_filled.shape}")
    
    # Step 3: Feature engineering
    df_featured = engineer_features(df_filled)
    print(f"After feature engineering - Shape: {df_featured.shape}")
    
    # Select PCA features FIRST
    pca_data, feature_names = select_pca_features(df_featured)
    print(f"PCA data shape: {pca_data.shape}")
    
    # THEN check for non-numeric values
    check_non_numeric_values(pca_data)
    
    # Step 4: PCA analysis
    pca_model, principal_df, scaler, scaled_data, final_features = perform_pca(pca_data)
    components_df, explained_var, cumulative_var = analyze_pca_results(
        pca_model, final_features, principal_df
    )
    
    # Step 5: Combine results and save
    for col in principal_df.columns:
        df_featured[col] = principal_df[col]
    
    save_results(df_featured, principal_df)
    print_summary(df, df_featured)
    
    print("\n🎉 Pipeline completed successfully!")
    return {
        'original_df': df,
        'cleaned_df': df_featured,
        'pca_data': pca_data,
        'principal_df': principal_df,
        'pca_model': pca_model,
        'components_df': components_df,
        'explained_variance': explained_var,
        'cumulative_variance': cumulative_var,
        'final_features': final_features
    }

if __name__ == "__main__":
    results = main()