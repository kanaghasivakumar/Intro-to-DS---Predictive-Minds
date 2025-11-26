import pandas as pd
import numpy as np
from config.config import PRICE_COLUMNS, DATE_COLUMNS
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def clean_data(df):
    print("Cleaning data")
    df_clean = df.copy()
    
    for col in PRICE_COLUMNS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace('[\\$,]', '', regex=True).astype(float)
    
    if 'host_response_rate' in df_clean.columns:
        df_clean['host_response_rate'] = df_clean['host_response_rate'].str.rstrip('%').astype(float)
    
    for col in DATE_COLUMNS:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    if 'host_since' in df_clean.columns:
        df_clean['host_experience_years'] = (pd.Timestamp.now() - df_clean['host_since']).dt.days / 365.25
    
    if 'amenities' in df_clean.columns:
        df_clean['amenities_count'] = df_clean['amenities'].str.count(',') + 1
        df_clean['amenities_count'] = df_clean['amenities_count'].fillna(0)
    
    print("Data cleaning completed!")
    return df_clean

def handle_missing_values(df):
    print("Handling missing values")
    df_filled = df.copy()
    
    print("\n=== ANALYZING MISSINGNESS ===")
    missing_percent = (df_filled.isnull().sum() / len(df_filled)) * 100
    high_missing = missing_percent[missing_percent > 40]
    moderate_missing = missing_percent[(missing_percent >= 10) & (missing_percent <= 40)]
    low_missing = missing_percent[(missing_percent > 0) & (missing_percent < 10)]
    
    print(f"High missing columns (>40%): {len(high_missing)}")
    print(f"Moderate missing columns (10-40%): {len(moderate_missing)}")
    print(f"Low missing columns (<10%): {len(low_missing)}")
    
    total_numeric_values_filled = 0
    total_categorical_values_filled = 0
    numeric_cols_imputed = []
    categorical_cols_filled = []
    mode_filled = 0
    unknown_filled = 0
    columns_dropped = []
    missing_indicators_created = []
    
    print("\n=== TIER 1: DROPPING HIGH MISSING COLUMNS (>40%) ===")
    for col, percent in high_missing.items():
        if col in df_filled.columns:
            df_filled = df_filled.drop(columns=[col])
            columns_dropped.append((col, percent))
            print(f"   DROPPED: {col} ({percent:.1f}% missing)")
    
    print(f"   Total columns dropped: {len(columns_dropped)}")
    
    print("\n=== TIER 2: CREATING MISSING INDICATORS (10-40%) ===")
    for col, percent in moderate_missing.items():
        if col in df_filled.columns:
            indicator_name = f'has_{col}'
            df_filled[indicator_name] = ~df_filled[col].isnull()
            missing_indicators_created.append((col, percent))
            print(f"   CREATED INDICATOR: {indicator_name} for {col} ({percent:.1f}% missing)")
    
    print(f"   Total missing indicators created: {len(missing_indicators_created)}")
    
    print("\n=== TIER 3: reviews_per_month SPECIAL HANDLING ===")
    if 'reviews_per_month' in df_filled.columns:
        missing_count = df_filled['reviews_per_month'].isnull().sum()
        if missing_count > 0:
            zero_review_count = (df_filled['reviews_per_month'].isnull() & 
                               (df_filled['number_of_reviews'].fillna(0) == 0)).sum()
            
            df_filled['reviews_per_month'] = df_filled['reviews_per_month'].fillna(0)
            total_numeric_values_filled += missing_count
            numeric_cols_imputed.append('reviews_per_month')
            print(f"   Filled {missing_count} missing reviews_per_month with 0")
            print(f"   ({zero_review_count} had zero reviews)")
    
    print("\n=== TIER 4: STANDARD IMPUTATION FOR REMAINING COLUMNS ===")
    
    numeric_columns = df_filled.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df_filled.select_dtypes(include=['object']).columns.tolist()
    
    print(f"   Numeric columns remaining: {len(numeric_columns)}")
    print(f"   Categorical columns remaining: {len(categorical_columns)}")
    
    numeric_cols_to_impute = []
    for col in numeric_columns:
        if col not in ['reviews_per_month'] and df_filled[col].isnull().sum() > 0 and df_filled[col].notna().sum() > 0:
            numeric_cols_to_impute.append(col)
    
    if numeric_cols_to_impute:
        pre_impute_missing = df_filled[numeric_cols_to_impute].isnull().sum().sum()
        numeric_imputer = SimpleImputer(strategy='median')
        df_filled[numeric_cols_to_impute] = numeric_imputer.fit_transform(df_filled[numeric_cols_to_impute])
        total_numeric_values_filled += pre_impute_missing
        numeric_cols_imputed.extend(numeric_cols_to_impute)
        print(f"   Imputed {len(numeric_cols_to_impute)} numeric columns")
        print(f"   Filled {pre_impute_missing} numeric values with median")
    
    completely_empty_numeric = []
    for col in numeric_columns:
        if col in df_filled.columns and df_filled[col].isnull().all():
            completely_empty_numeric.append(col)
    
    if completely_empty_numeric:
        df_filled = df_filled.drop(columns=completely_empty_numeric)
        print(f"   Dropped {len(completely_empty_numeric)} completely empty numeric columns: {completely_empty_numeric}")
    
    categorical_cols_filled_list = []
    for col in categorical_columns:
        if col in df_filled.columns and df_filled[col].isnull().sum() > 0:
            missing_count = df_filled[col].isnull().sum()
            if df_filled[col].nunique() < 50: 
                if not df_filled[col].mode().empty:
                    df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])
                    mode_filled += missing_count
                else:
                    df_filled[col] = df_filled[col].fillna('Unknown')
                    unknown_filled += missing_count
            else: 
                df_filled[col] = df_filled[col].fillna('Unknown')
                unknown_filled += missing_count
            categorical_cols_filled_list.append(col)
            total_categorical_values_filled += missing_count
    
    print(f"   Filled {len(categorical_cols_filled_list)} categorical columns")
    print(f"   Filled {total_categorical_values_filled} categorical values (Mode: {mode_filled}, Unknown: {unknown_filled})")
    
    print("\n=== FINAL SUMMARY ===")
    print(f"   Columns dropped: {len(columns_dropped)}")
    print(f"   Missing indicators created: {len(missing_indicators_created)}")
    print(f"   Numeric columns imputed: {len(numeric_cols_imputed)}")
    print(f"   Categorical columns filled: {len(categorical_cols_filled_list)}")
    print(f"   Total numeric values filled: {total_numeric_values_filled}")
    print(f"   Total categorical values filled: {total_categorical_values_filled}")
    print(f"   Grand total values imputed: {total_numeric_values_filled + total_categorical_values_filled}")
    
    remaining_missing = df_filled.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"   {remaining_missing} missing values remain")
    
    print(f"Final dataset shape: {df_filled.shape}")
    print("Missing values handled")
    return df_filled