import re
import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import os

def extract_amenity_features(df):
    """
    Extract specific amenity indicators from amenities column
    """
    print("=== EXTRACTING AMENITY FEATURES ===")
    
    amenity_mapping = {
        'has_pool': r'pool|Pool',
        'has_parking': r'parking|Parking',
        'has_kitchen': r'kitchen|Kitchen',
        'has_wifi': r'wifi|WiFi|Wifi',
        'has_ac': r'air conditioning|AC|A/C',
        'has_laundry': r'washer|dryer|laundry|Laundry',
        'has_gym': r'gym|Gym|fitness|Fitness',
        'has_breakfast': r'breakfast|Breakfast',
        'has_pet_friendly': r'pet|Pet',
        'has_balcony': r'balcony|Balcony|patio|Patio'
    }
    
    for amenity, pattern in amenity_mapping.items():
        df[amenity] = df['amenities'].str.contains(pattern, na=False).astype(int)
        print(f"{amenity}: {df[amenity].sum()} listings")
    
    return df

def analyze_amenity_premiums(df):
    """
    Calculate price premiums for specific amenities
    """
    print("=== AMENITY PREMIUM ANALYSIS ===")
    
    # Use EXACTLY the same control features as neighborhood analysis
    control_features = [
        'accommodates', 'bedrooms', 'bathrooms', 'beds',
        'minimum_nights', 'availability_30',
        'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified',
        'instant_bookable', 'amenities_count'
    ]
    
    # Convert binary features from t/f to 1/0 (EXACTLY like neighborhood analysis)
    binary_features = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable']
    for feature in binary_features:
        if feature in df.columns:
            df[feature] = df[feature].map({'t': 1, 'f': 0}).fillna(0)
            print(f"Converted {feature}: {df[feature].value_counts().to_dict()}")
    
    # ONLY include the actual amenity features we extracted, not the missing value indicators
    amenity_features = ['has_pool', 'has_parking', 'has_kitchen', 'has_wifi', 'has_ac', 
                       'has_laundry', 'has_gym', 'has_breakfast', 'has_pet_friendly', 'has_balcony']
    
    print(f"Found {len(amenity_features)} amenity features: {amenity_features}")
    
    features = control_features + amenity_features
    
    available_features = [f for f in features if f in df.columns]
    print(f"Using {len(available_features)} features for analysis")
    
    # Create clean dataset (EXACTLY like neighborhood analysis)
    X = df[available_features].copy()
    y = np.log(df['price'])
    
    # Convert all boolean columns to int64 (EXACTLY like neighborhood analysis)
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype('int64')
            print(f"Converted {col} from bool to int64")
    
    # Remove rows with any NaN values (EXACTLY like neighborhood analysis)
    valid_indices = X.notna().all(axis=1) & y.notna()
    X_clean = X[valid_indices]
    y_clean = y[valid_indices]
    
    print(f"Final data shape: X={X_clean.shape}, y={y_clean.shape}")
    
    if X_clean.shape[0] == 0:
        print("ERROR: No valid data remaining after cleaning")
        return None, None
    
    X_sm = sm.add_constant(X_clean)
    model = sm.OLS(y_clean, X_sm).fit()
    
    # Extract amenity premiums
    amenity_premiums = []
    for amenity in amenity_features:
        if amenity in model.params:
            coef = model.params[amenity]
            pval = model.pvalues[amenity]
            premium_pct = (np.exp(coef) - 1) * 100
            
            amenity_premiums.append({
                'amenity': amenity.replace('has_', ''),
                'premium_percent': premium_pct,
                'premium_multiplier': np.exp(coef),
                'p_value': pval,
                'significant': pval < 0.05,
                'count': df[amenity].sum()
            })
    
    amenity_df = pd.DataFrame(amenity_premiums)
    
    # Visualization
    if len(amenity_df) > 0:
        plt.figure(figsize=(10, 6))
        significant_amenities = amenity_df[amenity_df['significant']].sort_values('premium_percent')
        
        if len(significant_amenities) > 0:
            plt.barh(significant_amenities['amenity'], significant_amenities['premium_percent'])
            plt.xlabel('Price Premium (%)')
            plt.title('Amenity Price Premiums (Controlling for Property Characteristics)')
            plt.tight_layout()
            plt.savefig('results/amenity_premiums/amenity_premiums.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved visualization with {len(significant_amenities)} significant amenities")
    
    return amenity_df, model

if __name__ == "__main__":
    print("Loading pre-processed data...")
    df = pd.read_csv('../Dataset Processed/la_airbnb_cleaned_and_missing_values_handled.csv')
    
    print(f"Data loaded: {df.shape}")
    print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    
    os.makedirs('results/amenity_premiums', exist_ok=True)
    
    # Extract amenities first
    df_with_amenities = extract_amenity_features(df)
    
    # Run analysis
    amenity_results, model = analyze_amenity_premiums(df_with_amenities)
    
    if amenity_results is not None and len(amenity_results) > 0:
        print("\n" + "="*60)
        print("AMENITY PREMIUM ANALYSIS RESULTS")
        print("="*60)
        print(f"Total amenities analyzed: {len(amenity_results)}")
        print(f"Statistically significant amenities: {amenity_results['significant'].sum()}")
        
        print("\nTOP AMENITY PREMIUMS:")
        significant_amenities = amenity_results[amenity_results['significant']].nlargest(10, 'premium_percent')
        for _, row in significant_amenities.iterrows():
            print(f"  {row['amenity']}: +{row['premium_percent']:.1f}% (n={row['count']})")
        
        print("\nAMENITIES WITH NEGATIVE EFFECT:")
        negative_amenities = amenity_results[amenity_results['premium_percent'] < 0]
        for _, row in negative_amenities.iterrows():
            significance = "***" if row['significant'] else ""
            print(f"  {row['amenity']}: {row['premium_percent']:.1f}% {significance}")
        
        print(f"\nVisualization saved to: results/amenity_premiums/amenity_premiums.png")
        print("Amenity analysis complete!")
        
        # Show model summary
        print("\nMODEL SUMMARY:")
        print(f"R-squared: {model.rsquared:.3f}")
        print(f"Adjusted R-squared: {model.rsquared_adj:.3f}")
        
    else:
        print("Analysis failed or no amenity results found")