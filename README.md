# Intro-to-DS---Predictive-Minds
This will be looking at LA AirBnB data and doing cool :B stuff with it

Updating README after PCA:

    Success Summary:
    ✅ Data Loading & Exploration: 45,421 listings with 79 original columns

    ✅ Data Cleaning: Added 2 new features (host_experience_years, amenities_count)

    ✅ Missing Values Handled: 17 numeric columns imputed, 1 dropped, 21 categorical filled

    ✅ Feature Engineering: Expanded to 97 columns with encoded features

    ✅ PCA Analysis:

    Started with 42 features for PCA

    Removed 1 problematic text column (neighborhood_overview)

    Selected 29 principal components explaining 95.44% of variance

    All components saved successfully

Key PCA Insights:

Top Principal Components Explained:

PC1 (13.74% variance) - "Review Quality"

Dominated by all review scores (rating, accuracy, value, communication)

Higher values indicate better-reviewed properties

PC2 (9.98% variance) - "Property Size/Capacity"

Positive: accommodates, bedrooms, bathrooms, beds

Negative: private rooms (smaller properties)

Measures property size and guest capacity

PC3 (8.07% variance) - "Availability"

High availability across 30, 60, 90 days

Negative correlation with entire home/apartments (they book faster)

PC4 (5.39% variance) - "Popularity & Host Quality"

Reviews per month, number of reviews

Superhost status, amenities count

Measures listing popularity and host reputation

PC5 (4.54% variance) - "Location & Property Type"

Strong neighborhood effects (Hollywood vs Others)

Private room vs entire home patterns