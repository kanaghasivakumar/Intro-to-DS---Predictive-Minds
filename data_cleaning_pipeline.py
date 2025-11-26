from src.data_loader import load_data, explore_data
from src.data_cleaner import clean_data, handle_missing_values
from src.utils import save_results, print_summary

def run_data_cleaning_pipeline():
    print("Starting LA Airbnb Data Cleaning Pipeline\n")
    
    df = load_data()
    if df is None:
        return None
    
    explore_data(df)
    df_clean = clean_data(df)
    df_filled = handle_missing_values(df_clean)
    
    save_results(df_clean, df_filled)
    print_summary(df, df_filled)
    
    print("\nData cleaning pipeline completed successfully")
    return {'original_df': df, 'cleaned_df': df_filled}

if __name__ == "__main__": 
    results = run_data_cleaning_pipeline()