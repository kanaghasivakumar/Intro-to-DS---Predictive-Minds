# data_cleaner.py - FIXED VERSION
import pandas as pd
import numpy as np
from config import PRICE_COLUMNS, DATE_COLUMNS
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def clean_data(df):
    """Step 1: Clean the raw data"""
    print("🧹 Cleaning data...")
    df_clean = df.copy()
    
    # Clean price columns - FIXED escape sequence
    for col in PRICE_COLUMNS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace('[\\$,]', '', regex=True).astype(float)
    
    # Clean percentage columns
    if 'host_response_rate' in df_clean.columns:
        df_clean['host_response_rate'] = df_clean['host_response_rate'].str.rstrip('%').astype(float)
    
    # Convert date columns
    for col in DATE_COLUMNS:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    # Extract features from dates
    if 'host_since' in df_clean.columns:
        df_clean['host_experience_years'] = (pd.Timestamp.now() - df_clean['host_since']).dt.days / 365.25
    
    # Count amenities
    if 'amenities' in df_clean.columns:
        df_clean['amenities_count'] = df_clean['amenities'].str.count(',') + 1
        df_clean['amenities_count'] = df_clean['amenities_count'].fillna(0)
    
    print("✅ Data cleaning completed!")
    return df_clean

def handle_missing_values(df):
    """Step 2: Handle missing values - FIXED VERSION"""
    print("🔍 Handling missing values...")
    df_filled = df.copy()
    
    # Separate numeric and categorical columns
    numeric_columns = df_filled.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df_filled.select_dtypes(include=['object']).columns.tolist()
    
    # Handle numeric columns - only impute columns with some non-null values
    numeric_cols_to_impute = []
    for col in numeric_columns:
        if df_filled[col].isnull().sum() > 0 and df_filled[col].notna().sum() > 0:
            numeric_cols_to_impute.append(col)
    
    if numeric_cols_to_impute:
        numeric_imputer = SimpleImputer(strategy='median')
        df_filled[numeric_cols_to_impute] = numeric_imputer.fit_transform(df_filled[numeric_cols_to_impute])
        print(f"   Imputed {len(numeric_cols_to_impute)} numeric columns")
    
    # Drop numeric columns that are completely empty
    completely_empty_numeric = []
    for col in numeric_columns:
        if df_filled[col].isnull().all():
            completely_empty_numeric.append(col)
    
    if completely_empty_numeric:
        df_filled = df_filled.drop(columns=completely_empty_numeric)
        print(f"   Dropped {len(completely_empty_numeric)} completely empty numeric columns: {completely_empty_numeric}")
    
    # Handle categorical columns
    categorical_cols_filled = 0
    for col in categorical_columns:
        if df_filled[col].isnull().sum() > 0:
            if df_filled[col].nunique() < 50:  # For low-cardinality columns
                if not df_filled[col].mode().empty:
                    df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])
                else:
                    df_filled[col] = df_filled[col].fillna('Unknown')
            else:  # For high-cardinality columns
                df_filled[col] = df_filled[col].fillna('Unknown')
            categorical_cols_filled += 1
    
    print(f"   Filled {categorical_cols_filled} categorical columns")
    
    # Final check for any remaining missing values
    remaining_missing = df_filled.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"   ⚠️  {remaining_missing} missing values remain - these will be handled later")
    
    print("✅ Missing values handled!")
    return df_filled