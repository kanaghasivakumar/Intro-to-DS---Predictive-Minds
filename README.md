# LA Airbnb Data Analysis

Comprehensive analysis of Los Angeles Airbnb listings using Principal Component Analysis (PCA).

## Project Structure

- `main.py` - Main execution pipeline
- `config.py` - Configuration constants and paths
- `data_loader.py` - Data loading and exploration
- `data_cleaner.py` - Data cleaning and missing value handling
- `feature_engineer.py` - Feature engineering and selection
- `pca_analyzer.py` - PCA analysis and component selection
- `visualizations.py` - Comprehensive visualization suite
- `run_visualizations.py` - Run all visualizations

## Key Findings

- **29 Principal Components** explain **95.44%** of variance
- **PC1 (13.74%)**: Review Quality (all review scores)
- **PC2 (9.98%)**: Property Size (accommodates, bedrooms, bathrooms)
- **PC3 (8.07%)**: Availability (30/60/90 day availability)
- **PC4 (5.39%)**: Popularity (reviews per month, superhost status)

## Usage

1. Run the full pipeline:
   python main.py
2. Generate visualizations:
   python run_visualizations.py

## Guide

🏆 PC1 - "Review Quality" (13.74%)
What it represents: How well-reviewed a property is

High positive loadings: All review scores (rating, accuracy, value, communication)

Interpretation: Properties with high PC1 scores have excellent reviews across all categories

🏠 PC2 - "Property Size" (9.98%)
What it represents: Physical capacity of the property

Positive: accommodates, bedrooms, bathrooms, beds

Negative: Private rooms (smaller properties)

Interpretation: High PC2 = large properties that can host more guests

📅 PC3 - "Availability" (8.07%)
What it represents: How available the property is

Positive: Availability across 30, 60, 90 days

Negative: Entire homes/apartments (they book faster)

Interpretation: High PC3 = properties that are frequently available

⭐ PC4 - "Popularity & Host Quality" (5.39%)
What it represents: How popular and well-managed the listing is

Positive: reviews_per_month, number_of_reviews, superhost status

Interpretation: High PC4 = frequently reviewed properties with reputable hosts

🎯 PC5-10 (26.85% variance) - "Neighborhood & Property Type Effects"
PC5 (4.54%) - "Location vs Property Type"

High: neighborhood_Other, room_type_Private room

Low: neighborhood_Hollywood, room_type_Entire home/apt

Meaning: Distinguishes private rooms in less popular areas vs entire homes in Hollywood

PC6 (3.71%) - "Host Experience vs Property Type"

High: room_type_Entire home/apt, host_experience_years

Low: room_type_Private room

Meaning: Experienced hosts tend to offer entire homes

PC7 (3.27%) - "Instant Booking Dynamics"

High: host_experience_years

Low: instant_bookable

Meaning: Experienced hosts are less likely to use instant booking

PC8 (3.03%) - "Hotel vs Long-term Stays"

High: room_type_Hotel room, maximum_nights

Meaning: Hotel rooms allow longer maximum stays

PC9 (2.67%) - "Host Verification Patterns"

High: host_identity_verified, neighborhood_Santa Monica

Meaning: Santa Monica has more verified hosts

PC10 (2.62%) - "Hollywood Shared Rooms"

High: neighborhood_Hollywood, room_type_Shared room

Meaning: Hollywood has more shared room listings

🗺️ PC11-19 (27.53% variance) - "Neighborhood Micro-Patterns"
These components capture very specific geographic patterns:

PC11: Long Beach vs Downtown vs Venice dynamics

PC12: Downtown vs Pasadena contrast

PC13: Beverly Hills vs Alhambra differences

PC14: Santa Monica dominance

PC15: Hollywood Hills uniqueness

PC16: Pasadena-Santa Monica relationship

PC17: Alhambra-West Hollywood opposition

PC18: Shared rooms in specific areas

PC19: More geographic fine-tuning

⚙️ PC20-29 (21.88% variance) - "Technical & Minor Effects"
PC20 (2.21%): Host verification + stay duration patterns

PC21 (2.12%): Instant booking + price relationships

PC22 (2.05%): Minimum nights + superhost correlation

PC23 (1.99%): Maximum vs minimum nights dynamics

PC24 (1.78%): Pure price effects

PC25 (1.60%): Host experience patterns

PC26 (1.41%): Amenities vs superhost tradeoff

PC27 (1.17%): Long-term availability (365 days)

PC28 (0.95%): Location rating specificity

PC29 (0.90%): Check-in vs cleanliness rating balance
