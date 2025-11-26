import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_neighborhood_effects(df):
    print("=== NEIGHBORHOOD PRICE ANALYSIS ===")
    
    # First, let's create the neighborhood dummy variables
    print("Creating neighborhood dummy variables...")
    top_neighborhoods = df['neighbourhood_cleansed'].value_counts().head(10).index
    df_analysis = df.copy()
    df_analysis['neighbourhood_group'] = df_analysis['neighbourhood_cleansed'].apply(
        lambda x: x if x in top_neighborhoods else 'Other'
    )
    neighborhood_dummies = pd.get_dummies(df_analysis['neighbourhood_group'], prefix='neighborhood')
    df_analysis = pd.concat([df_analysis, neighborhood_dummies], axis=1)
    
    # Convert binary features from t/f to 1/0
    binary_features = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable']
    for feature in binary_features:
        if feature in df_analysis.columns:
            df_analysis[feature] = df_analysis[feature].map({'t': 1, 'f': 0}).fillna(0)
            print(f"Converted {feature}: {df_analysis[feature].value_counts().to_dict()}")
    
    control_features = [
        'accommodates', 'bedrooms', 'bathrooms', 'beds',
        'minimum_nights', 'availability_30',
        'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified',
        'instant_bookable', 'amenities_count'
    ]
    
    neighborhood_features = [col for col in df_analysis.columns if col.startswith('neighborhood_')]
    
    print(f"Found {len(neighborhood_features)} neighborhood features: {neighborhood_features}")
    
    features = control_features + neighborhood_features
    
    available_features = [f for f in features if f in df_analysis.columns]
    print(f"Using {len(available_features)} features for analysis")
    
    # Create clean dataset
    X = df_analysis[available_features].copy()
    y = np.log(df_analysis['price'])
    
    # Convert all boolean columns to int64
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype('int64')
            print(f"Converted {col} from bool to int64")
    
    # Debug: Check data types
    print("\nData types in X:")
    for col in X.columns:
        print(f"  {col}: {X[col].dtype}")
    
    # Remove rows with any NaN values
    valid_indices = X.notna().all(axis=1) & y.notna()
    X_clean = X[valid_indices]
    y_clean = y[valid_indices]
    
    print(f"Final data shape: X={X_clean.shape}, y={y_clean.shape}")
    
    if X_clean.shape[0] == 0:
        print("ERROR: No valid data remaining after cleaning")
        return None, None
    
    # Final check - ensure all data is numeric
    print("Final data types:")
    for col in X_clean.columns:
        print(f"  {col}: {X_clean[col].dtype}")
    
    X_sm = sm.add_constant(X_clean)
    model = sm.OLS(y_clean, X_sm).fit()
    
    neighborhood_coefs = []
    for feature in neighborhood_features:
        if feature in model.params:
            coef = model.params[feature]
            pval = model.pvalues[feature]
            neighborhood_coefs.append({
                'neighborhood': feature.replace('neighborhood_', ''),
                'premium_multiplier': np.exp(coef),
                'premium_percent': (np.exp(coef) - 1) * 100,
                'p_value': pval,
                'significant': pval < 0.05
            })
    
    neighborhood_df = pd.DataFrame(neighborhood_coefs)
    
    if len(neighborhood_df) > 0:
        plt.figure(figsize=(12, 8))
        significant_df = neighborhood_df[neighborhood_df['significant']].sort_values('premium_percent')
        
        if len(significant_df) > 0:
            plt.barh(significant_df['neighborhood'], significant_df['premium_percent'])
            plt.xlabel('Price Premium (%)')
            plt.title('Neighborhood Price Premiums (Controlling for Property Characteristics)')
            plt.tight_layout()
            plt.savefig('results/neighborhood_effects/neighborhood_premiums.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved visualization with {len(significant_df)} significant neighborhoods")
        else:
            print("No significant neighborhoods found for visualization")
    else:
        print("No neighborhood coefficients found in model")
    
    return neighborhood_df, model

if __name__ == "__main__":
    print("Loading pre-processed data...")
    df = pd.read_csv('../Dataset Processed/la_airbnb_cleaned_and_missing_values_handled.csv')
    
    print(f"Data loaded: {df.shape}")
    print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    
    os.makedirs('results/neighborhood_effects', exist_ok=True)
    
    neighborhood_results, model = analyze_neighborhood_effects(df)
    
    if neighborhood_results is not None and len(neighborhood_results) > 0:
        print("="*60)
        print("NEIGHBORHOOD ANALYSIS RESULTS")
        print("="*60)
        print(f"Total neighborhoods analyzed: {len(neighborhood_results)}")
        print(f"Statistically significant neighborhoods: {neighborhood_results['significant'].sum()}")
        
        print("\nTOP 5 MOST EXPENSIVE NEIGHBORHOODS:")
        top_5 = neighborhood_results.nlargest(5, 'premium_percent')
        for _, row in top_5.iterrows():
            significance = "***" if row['significant'] else ""
            print(f"  {row['neighborhood']}: +{row['premium_percent']:.1f}% {significance}")
        
        print("\nTOP 5 LEAST EXPENSIVE NEIGHBORHOODS:")
        bottom_5 = neighborhood_results.nsmallest(5, 'premium_percent')
        for _, row in bottom_5.iterrows():
            significance = "***" if row['significant'] else ""
            print(f"  {row['neighborhood']}: {row['premium_percent']:.1f}% {significance}")
        
        print(f"\nVisualization saved to: results/neighborhood_effects/neighborhood_premiums.png")
        print("Neighborhood analysis complete!")
        
        # Show model summary
        print("\nMODEL SUMMARY:")
        print(f"R-squared: {model.rsquared:.3f}")
        print(f"Adjusted R-squared: {model.rsquared_adj:.3f}")
        
    else:
        print("Analysis failed or no neighborhood results found")


# print("INTERPRETATION:")
# print("• The model explains 50.7% of Airbnb price variation in LA")
# print("• Neighborhood and property characteristics are MAJOR price drivers")  
# print("• The remaining 49.3% is due to: photos, host reputation, temporary demand, unique amenities, etc.")
# print("• This is STRONG evidence for systematic neighborhood pricing effects")