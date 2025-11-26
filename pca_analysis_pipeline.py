from src.data_loader import load_data
from src.data_cleaner import clean_data, handle_missing_values
from src.feature_engineer import engineer_features, select_pca_features
from src.pca_analyzer import perform_pca, analyze_pca_results
from src.debug_utils import check_non_numeric_values

def run_pca_analysis_pipeline():
    print("Starting LA Airbnb PCA Analysis Pipeline\n")
    
    df = load_data()
    if df is None:
        return None
    
    df_clean = clean_data(df)
    df_filled = handle_missing_values(df_clean)
    df_featured = engineer_features(df_filled)
    
    pca_data, feature_names = select_pca_features(df_featured)
    check_non_numeric_values(pca_data)
    
    pca_model, principal_df, scaler, scaled_data, final_features = perform_pca(pca_data)
    components_df, explained_var, cumulative_var = analyze_pca_results(
        pca_model, final_features, principal_df
    )
    
    for col in principal_df.columns:
        df_featured[col] = principal_df[col]
    
    print("\nPCA analysis pipeline completed successfully")
    return {
        'cleaned_df': df_featured,
        'pca_model': pca_model,
        'components_df': components_df,
        'explained_variance': explained_var
    }

if __name__ == "__main__":
    results = run_pca_analysis_pipeline()