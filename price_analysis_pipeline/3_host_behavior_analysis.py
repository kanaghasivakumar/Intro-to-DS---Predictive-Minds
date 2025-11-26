import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import os
import statsmodels.api as sm

def analyze_host_behavior(df):
    """
    Analyze multi-listing host clustering and pricing behavior
    """
    print("=== MULTI-LISTING HOST ANALYSIS ===")
    
    # Define multi-listing hosts (hosts with >1 listing)
    df['is_multi_lister'] = df['calculated_host_listings_count'] > 1
    df['is_professional_host'] = df['calculated_host_listings_count'] > 5
    
    # Analyze geographic clustering
    neighborhood_host_counts = df.groupby('neighbourhood_cleansed').agg({
        'host_id': 'nunique',
        'calculated_host_listings_count': 'sum',
        'price': 'mean'
    }).reset_index()
    
    neighborhood_host_counts['listings_per_host'] = (
        neighborhood_host_counts['calculated_host_listings_count'] / 
        neighborhood_host_counts['host_id']
    )
    
    # Compare pricing behavior
    pricing_comparison = df.groupby('is_multi_lister').agg({
        'price': ['mean', 'median', 'count'],
        'review_scores_rating': 'mean',
        'availability_30': 'mean'
    }).round(2)
    
    # Professional vs individual host pricing by neighborhood
    # Focus on top neighborhoods for cleaner analysis
    top_neighborhoods = df['neighbourhood_cleansed'].value_counts().head(10).index
    df_top_neighborhoods = df[df['neighbourhood_cleansed'].isin(top_neighborhoods)]
    
    professional_pricing = df_top_neighborhoods.groupby(['neighbourhood_cleansed', 'is_professional_host']).agg({
        'price': 'median',
        'number_of_reviews': 'mean'
    }).unstack().round(2)
    
    # Statistical test for pricing differences
    multi_lister_prices = df[df['is_multi_lister']]['price']
    single_lister_prices = df[~df['is_multi_lister']]['price']
    
    t_stat, p_value = stats.ttest_ind(multi_lister_prices, single_lister_prices, equal_var=False)
    
    print(f"Multi-lister vs Single-lister price difference p-value: {p_value:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Price distribution by host type
    sns.boxplot(data=df, x='is_multi_lister', y='price', ax=axes[0, 0])
    axes[0, 0].set_title('Price Distribution: Multi-lister vs Single-lister')
    axes[0, 0].set_xlabel('Is Multi-lister')
    axes[0, 0].set_ylabel('Price ($)')
    
    # Geographic clustering
    top_neighborhoods_clustering = neighborhood_host_counts.nlargest(10, 'listings_per_host')
    sns.barplot(data=top_neighborhoods_clustering, x='listings_per_host', y='neighbourhood_cleansed', ax=axes[0, 1])
    axes[0, 1].set_title('Top Neighborhoods by Listings per Host')
    axes[0, 1].set_xlabel('Average Listings per Host')
    
    # Review scores comparison
    sns.boxplot(data=df, x='is_multi_lister', y='review_scores_rating', ax=axes[1, 0])
    axes[1, 0].set_title('Review Scores: Multi-lister vs Single-lister')
    axes[1, 0].set_xlabel('Is Multi-lister')
    axes[1, 0].set_ylabel('Review Score')
    
    # Availability comparison
    sns.boxplot(data=df, x='is_multi_lister', y='availability_30', ax=axes[1, 1])
    axes[1, 1].set_title('Availability: Multi-lister vs Single-lister')
    axes[1, 1].set_xlabel('Is Multi-lister')
    axes[1, 1].set_ylabel('Availability (30 days)')
    
    plt.tight_layout()
    plt.savefig('results/host_behavior/host_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional analysis: Price premium by host type controlling for neighborhood
    print("\n=== PRICE PREMIUM ANALYSIS CONTROLLING FOR NEIGHBORHOOD ===")
    
    # Create neighborhood dummies for regression
    top_neighborhoods_reg = df['neighbourhood_cleansed'].value_counts().head(10).index
    df_reg = df.copy()
    df_reg['neighbourhood_group'] = df_reg['neighbourhood_cleansed'].apply(
        lambda x: x if x in top_neighborhoods_reg else 'Other'
    )
    neighborhood_dummies = pd.get_dummies(df_reg['neighbourhood_group'], prefix='neighborhood')
    df_reg = pd.concat([df_reg, neighborhood_dummies], axis=1)
    
    # Control features
    control_features = [
        'accommodates', 'bedrooms', 'bathrooms', 'beds',
        'minimum_nights', 'availability_30',
        'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified',
        'instant_bookable', 'amenities_count'
    ]
    
    # Convert binary features
    binary_features = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable']
    for feature in binary_features:
        if feature in df_reg.columns and df_reg[feature].dtype == 'object':
            df_reg[feature] = df_reg[feature].map({'t': 1, 'f': 0}).fillna(0)
    
    neighborhood_features = [col for col in df_reg.columns if col.startswith('neighborhood_')]
    features = control_features + neighborhood_features + ['is_professional_host']
    
    X = df_reg[features].copy()
    y = np.log(df_reg['price'])
    
    # Convert boolean columns
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype('int64')
    
    # Remove missing values
    valid_indices = X.notna().all(axis=1) & y.notna()
    X_clean = X[valid_indices]
    y_clean = y[valid_indices]
    
    X_sm = sm.add_constant(X_clean)
    model = sm.OLS(y_clean, X_sm).fit()
    
    if 'is_professional_host' in model.params:
        coef = model.params['is_professional_host']
        pval = model.pvalues['is_professional_host']
        premium_pct = (np.exp(coef) - 1) * 100
        print(f"Professional host premium (after controlling for neighborhood/property): {premium_pct:.1f}%")
        print(f"Statistical significance: p = {pval:.4f}")
    
    return {
        'neighborhood_host_counts': neighborhood_host_counts,
        'pricing_comparison': pricing_comparison,
        'professional_pricing': professional_pricing,
        'price_difference_test': {'t_statistic': t_stat, 'p_value': p_value},
        'professional_host_premium': premium_pct if 'is_professional_host' in model.params else None
    }

if __name__ == "__main__":
    print("Loading pre-processed data...")
    df = pd.read_csv('../Dataset Processed/la_airbnb_cleaned_and_missing_values_handled.csv')
    
    print(f"Data loaded: {df.shape}")
    print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    
    os.makedirs('results/host_behavior', exist_ok=True)
    
    # Run analysis
    host_results = analyze_host_behavior(df)
    
    print("\n" + "="*60)
    print("HOST BEHAVIOR ANALYSIS RESULTS")
    print("="*60)
    
    print("\nPRICING COMPARISON:")
    print(host_results['pricing_comparison'])
    
    print(f"\nSTATISTICAL TEST:")
    print(f"Multi-lister vs Single-lister price difference: p = {host_results['price_difference_test']['p_value']:.4f}")
    
    if host_results['professional_host_premium'] is not None:
        print(f"\nPROFESSIONAL HOST PREMIUM:")
        print(f"Professional hosts charge {host_results['professional_host_premium']:.1f}% more after controlling for neighborhood and property characteristics")
    
    print("\nTOP 5 NEIGHBORHOODS BY HOST CONCENTRATION:")
    top_concentration = host_results['neighborhood_host_counts'].nlargest(5, 'listings_per_host')
    for _, row in top_concentration.iterrows():
        print(f"  {row['neighbourhood_cleansed']}: {row['listings_per_host']:.1f} listings per host")
    
    print(f"\nVisualization saved to: results/host_behavior/host_analysis.png")
    print("Host behavior analysis complete!")