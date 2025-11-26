import os
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util

def load_module_from_file(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def generate_final_report(neighborhood_results, amenity_results, host_results, integrated_results):
    print("="*80)
    print("FINAL ANALYSIS REPORT - LA AIRBNB PRICING DRIVERS")
    print("="*80)
    
    print("KEY FINDINGS SUMMARY")
    print("-" * 50)
    
    if neighborhood_results is not None and len(neighborhood_results) > 0:
        top_neighborhood = neighborhood_results.nlargest(1, 'premium_percent')
        print(f"STRONGEST NEIGHBORHOOD EFFECT: {top_neighborhood['neighborhood'].iloc[0]} (+{top_neighborhood['premium_percent'].iloc[0]:.1f}%)")
    
    if amenity_results is not None and len(amenity_results) > 0:
        top_amenity = amenity_results.nlargest(1, 'premium_percent')
        print(f"STRONGEST AMENITY EFFECT: {top_amenity['amenity'].iloc[0]} (+{top_amenity['premium_percent'].iloc[0]:.1f}%)")
    
    if host_results is not None and 'professional_host_premium' in host_results:
        print(f"PROFESSIONAL HOST PRICING: {host_results['professional_host_premium']:.1f}%")
    
    if integrated_results is not None:
        print(f"OVERALL MODEL EXPLANATORY POWER: {integrated_results['ols_model'].rsquared*100:.1f}%")
    
    print("\nDETAILED RESULTS")
    print("-" * 50)
    
    print("TOP 5 NEIGHBORHOOD PREMIUMS:")
    if neighborhood_results is not None:
        top_5_neighborhoods = neighborhood_results.nlargest(5, 'premium_percent')
        for _, row in top_5_neighborhoods.iterrows():
            print(f"  {row['neighborhood']}: +{row['premium_percent']:.1f}%")
    
    print("\nTOP 5 AMENITY PREMIUMS:")
    if amenity_results is not None:
        top_5_amenities = amenity_results.nlargest(5, 'premium_percent')
        for _, row in top_5_amenities.iterrows():
            print(f"  {row['amenity']}: +{row['premium_percent']:.1f}%")
    
    print("\nHOST BEHAVIOR INSIGHTS:")
    if host_results is not None:
        print(f"  Multi-lister price difference p-value: {host_results['price_difference_test']['p_value']:.4f}")
        if 'professional_host_premium' in host_results:
            print(f"  Professional host premium: {host_results['professional_host_premium']:.1f}%")
    
    print("\nINTEGRATED MODEL PERFORMANCE:")
    if integrated_results is not None:
        print(f"  R-squared: {integrated_results['ols_model'].rsquared:.3f}")
        print(f"  Top 3 features by importance:")
        top_features = integrated_results['feature_importance'].head(3)
        for _, row in top_features.iterrows():
            print(f"    {row['feature']}: {row['importance']:.3f}")

def main():
    os.makedirs('results/neighborhood_effects', exist_ok=True)
    os.makedirs('results/amenity_premiums', exist_ok=True)
    os.makedirs('results/host_behavior', exist_ok=True)
    os.makedirs('results/integrated_model', exist_ok=True)
    
    print("Loading data...")
    df = pd.read_csv('../Dataset Processed/la_airbnb_cleaned_and_missing_values_handled.csv')
    
    print(f"Data loaded: {df.shape}")
    print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    
    print("="*50)
    print("RUNNING COMPREHENSIVE PRICE ANALYSIS")
    print("="*50)
    
    neighborhood_module = load_module_from_file('1_neighborhood_analysis.py', 'neighborhood_analysis')
    neighborhood_results, neighborhood_model = neighborhood_module.analyze_neighborhood_effects(df)
    
    amenity_module = load_module_from_file('2_amenity_premium_analysis.py', 'amenity_analysis')
    df_with_amenities = amenity_module.extract_amenity_features(df)
    amenity_results, amenity_model = amenity_module.analyze_amenity_premiums(df_with_amenities)
    
    host_module = load_module_from_file('3_host_behavior_analysis.py', 'host_analysis')
    host_results = host_module.analyze_host_behavior(df)
    
    integrated_module = load_module_from_file('4_integrated_model.py', 'integrated_model')
    integrated_results = integrated_module.build_integrated_price_model(df)
    
    generate_final_report(neighborhood_results, amenity_results, host_results, integrated_results)
    
    print("="*50)
    print("ANALYSIS COMPLETE!")
    print("="*50)

if __name__ == "__main__":
    main()