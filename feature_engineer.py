import pandas as pd
from config import BINARY_FEATURES

def engineer_features(df):
    """Step 3: Create new features"""
    print("⚙️ Engineering features...")
    df_fe = df.copy()
    
    # Convert binary features
    for feature in BINARY_FEATURES:
        if feature in df_fe.columns:
            df_fe[feature] = df_fe[feature].map({'t': 1, 'f': 0}).fillna(0)
    
    # Room type encoding
    if 'room_type' in df_fe.columns:
        room_type_dummies = pd.get_dummies(df_fe['room_type'], prefix='room_type')
        df_fe = pd.concat([df_fe, room_type_dummies], axis=1)
    
    # Neighborhood encoding (top 10)
    if 'neighbourhood_cleansed' in df_fe.columns:
        top_neighborhoods = df_fe['neighbourhood_cleansed'].value_counts().head(10).index
        df_fe['neighbourhood_group'] = df_fe['neighbourhood_cleansed'].apply(
            lambda x: x if x in top_neighborhoods else 'Other'
        )
        neighborhood_dummies = pd.get_dummies(df_fe['neighbourhood_group'], prefix='neighborhood')
        df_fe = pd.concat([df_fe, neighborhood_dummies], axis=1)
    
    # Price per bedroom
    if all(col in df_fe.columns for col in ['price', 'bedrooms']):
        df_fe['price_per_bedroom'] = df_fe['price'] / df_fe['bedrooms'].replace(0, 1)
    
    print("✅ Feature engineering completed!")
    return df_fe

def select_pca_features(df):
    """Select features for PCA analysis"""
    print("🎯 Selecting features for PCA...")
    
    pca_features = [
        'price', 'accommodates', 'bathrooms', 'bedrooms', 'beds',
        'minimum_nights', 'maximum_nights', 'availability_30',
        'availability_60', 'availability_90', 'availability_365',
        'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy',
        'review_scores_cleanliness', 'review_scores_checkin',
        'review_scores_communication', 'review_scores_location',
        'review_scores_value', 'reviews_per_month',
        'amenities_count', 'host_experience_years'
    ]
    
    # Add binary features
    binary_cols = BINARY_FEATURES
    pca_features.extend([col for col in binary_cols if col in df.columns])
    
    # Add encoded features
    room_type_cols = [col for col in df.columns if col.startswith('room_type_')]
    pca_features.extend(room_type_cols)
    
    neighborhood_cols = [col for col in df.columns if col.startswith('neighborhood_')]
    pca_features.extend(neighborhood_cols)
    
    # Filter to existing features only
    pca_features = [feature for feature in pca_features if feature in df.columns]
    
    pca_df = df[pca_features].copy()
    
    print(f"✅ Selected {len(pca_features)} features for PCA")
    return pca_df, pca_features