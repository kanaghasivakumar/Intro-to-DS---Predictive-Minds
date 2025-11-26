import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import os

def build_integrated_price_model(df):
    """
    Build comprehensive price model incorporating all factors
    """
    print("=== INTEGRATED PRICE MODEL ===")
    
    # First, let's create the features we need that might not exist yet
    df_model = df.copy()
    
    # Create neighborhood dummies (like we did in neighborhood analysis)
    top_neighborhoods = df_model['neighbourhood_cleansed'].value_counts().head(10).index
    df_model['neighbourhood_group'] = df_model['neighbourhood_cleansed'].apply(
        lambda x: x if x in top_neighborhoods else 'Other'
    )
    neighborhood_dummies = pd.get_dummies(df_model['neighbourhood_group'], prefix='neighborhood')
    df_model = pd.concat([df_model, neighborhood_dummies], axis=1)
    
    # Create host behavior features (like we did in host analysis)
    df_model['is_multi_lister'] = df_model['calculated_host_listings_count'] > 1
    df_model['is_professional_host'] = df_model['calculated_host_listings_count'] > 5
    
    # Extract amenity features (like we did in amenity analysis)
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
        df_model[amenity] = df_model['amenities'].str.contains(pattern, na=False).astype(int)
    
    # Convert binary features
    binary_features = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable']
    for feature in binary_features:
        if feature in df_model.columns and df_model[feature].dtype == 'object':
            df_model[feature] = df_model[feature].map({'t': 1, 'f': 0}).fillna(0)
    
    # Feature sets - using features that actually exist in our data
    property_features = [
        'accommodates', 'bedrooms', 'bathrooms', 'beds',
        'minimum_nights', 'availability_30'
    ]
    
    host_features = [
        'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified',
        'instant_bookable', 'is_multi_lister', 'is_professional_host', 
        'host_experience_years', 'amenities_count'
    ]
    
    review_features = ['review_scores_rating', 'number_of_reviews', 'reviews_per_month']
    neighborhood_features = [col for col in df_model.columns if col.startswith('neighborhood_')]
    amenity_features = [col for col in df_model.columns if col.startswith('has_') and col in [
        'has_pool', 'has_parking', 'has_kitchen', 'has_wifi', 'has_ac', 
        'has_laundry', 'has_gym', 'has_breakfast', 'has_pet_friendly', 'has_balcony'
    ]]
    
    all_features = (property_features + neighborhood_features + 
                   amenity_features + host_features + review_features)
    
    # Filter to available features
    available_features = [f for f in all_features if f in df_model.columns]
    
    print(f"Using {len(available_features)} features for integrated model")
    
    X = df_model[available_features].copy()
    y = np.log(df_model['price'])
    
    # Convert boolean columns to int
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype('int64')
    
    # Remove rows with missing values
    valid_indices = X.notna().all(axis=1) & y.notna()
    X_clean = X[valid_indices]
    y_clean = y[valid_indices]
    
    print(f"Final data shape: X={X_clean.shape}, y={y_clean.shape}")
    
    # OLS model for interpretability
    X_sm = sm.add_constant(X_clean)
    full_model = sm.OLS(y_clean, X_sm).fit()
    
    print("=== MODEL SUMMARY ===")
    print(f"R-squared: {full_model.rsquared:.3f}")
    print(f"Adjusted R-squared: {full_model.rsquared_adj:.3f}")
    print(f"Number of observations: {full_model.nobs}")
    
    # Feature importance from Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_clean, y_clean)
    
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Top features visualization
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title('Top 15 Feature Importance for Airbnb Price Prediction')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig('results/integrated_model/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Show top coefficients from OLS
    print("\n=== TOP 10 POSITIVE EFFECTS ===")
    coefficients = pd.DataFrame({
        'feature': ['const'] + available_features,
        'coefficient': full_model.params,
        'p_value': full_model.pvalues
    })
    positive_effects = coefficients[
        (coefficients['feature'] != 'const') & 
        (coefficients['coefficient'] > 0) &
        (coefficients['p_value'] < 0.05)
    ].nlargest(10, 'coefficient')
    
    for _, row in positive_effects.iterrows():
        premium_pct = (np.exp(row['coefficient']) - 1) * 100
        print(f"  {row['feature']}: +{premium_pct:.1f}%")
    
    print("\n=== TOP 10 NEGATIVE EFFECTS ===")
    negative_effects = coefficients[
        (coefficients['feature'] != 'const') & 
        (coefficients['coefficient'] < 0) &
        (coefficients['p_value'] < 0.05)
    ].nsmallest(10, 'coefficient')
    
    for _, row in negative_effects.iterrows():
        discount_pct = (1 - np.exp(row['coefficient'])) * 100
        print(f"  {row['feature']}: -{discount_pct:.1f}%")
    
    return {
        'ols_model': full_model,
        'feature_importance': feature_importance,
        'rf_model': rf,
        'coefficients': coefficients
    }

if __name__ == "__main__":
    print("Loading pre-processed data...")
    df = pd.read_csv('../Dataset Processed/la_airbnb_cleaned_and_missing_values_handled.csv')
    
    print(f"Data loaded: {df.shape}")
    print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    
    os.makedirs('results/integrated_model', exist_ok=True)
    
    # Run integrated analysis
    results = build_integrated_price_model(df)
    
    print("\n" + "="*60)
    print("INTEGRATED MODEL RESULTS")
    print("="*60)
    
    print(f"\nModel Performance: R² = {results['ols_model'].rsquared:.3f}")
    print(f"This means the model explains {results['ols_model'].rsquared*100:.1f}% of price variation")
    
    print(f"\nTop 5 Most Important Features:")
    for _, row in results['feature_importance'].head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    print(f"\nVisualization saved to: results/integrated_model/feature_importance.png")
    print("Integrated analysis complete!")