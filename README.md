# LA Airbnb Data Analysis Project

A comprehensive analysis of Los Angeles Airbnb listings examining neighborhood price effects, amenity premiums, and host behavior patterns.

## Research Questions

1. **Neighborhood Effects**: Are certain neighborhoods systematically higher-priced after controlling for property characteristics?
2. **Amenity Premiums**: Do amenities like pools and parking add significant price premiums?
3. **Host Behavior**: Are multi-listing hosts clustering in high-tourism areas and pricing more aggressively?

## Project Structure

```
Intro-to-DS---Predictive-Minds/
├── config/                          # Configuration files
│   └── config.py
├── src/                             # Reusable modules
│   ├── data_loader.py
│   ├── data_cleaner.py
│   ├── feature_engineer.py
│   ├── pca_analyzer.py
│   ├── debug_utils.py
│   └── utils.py
├── price_analysis_pipeline/         # Main analysis pipelines
│   ├── 1_neighborhood_analysis.py
│   ├── 2_amenity_premium_analysis.py
│   ├── 3_host_behavior_analysis.py
│   ├── 4_integrated_model.py
│   └── main_price_analysis.py
├── data_cleaning_pipeline.py        # Data preprocessing pipeline
├── pca_analysis_pipeline.py         # PCA dimensionality reduction
├── main.py                          # Legacy main file
└── Dataset Processed/               # Cleaned datasets (local)
```

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Complete Analysis Pipeline
```bash
cd price_analysis_pipeline
python main_price_analysis.py
```

### Run Individual Analyses
```bash
# 1. Neighborhood price effects
python 1_neighborhood_analysis.py

# 2. Amenity price premiums  
python 2_amenity_premium_analysis.py

# 3. Multi-listing host behavior
python 3_host_behavior_analysis.py

# 4. Integrated price model
python 4_integrated_model.py
```

## Key Findings

### Neighborhood Price Premiums
- **Venice**: +78.9% premium
- **Santa Monica**: +73.9% premium
- **West Hollywood**: +69.5% premium
- Location explains most price variation

### Amenity Value-Add
- **Pool**: +23.1% premium
- **Gym**: +15.2% premium  
- **Breakfast**: +14.5% premium
- Basic amenities (wifi, kitchen) show no premium

### Host Behavior Patterns
- Professional hosts charge **-10.5% less** after controls
- Cluster in non-premium, high-density areas
- Optimize for occupancy over price premiums

### Model Performance
- **R² = 0.554** - Explains 55.4% of price variation
- Top features: bedrooms, bathrooms, accommodation capacity

## Pipeline Details

### Data Cleaning Pipeline
```bash
python data_cleaning_pipeline.py
```
- Loads raw Airbnb data
- Cleans price columns and dates
- Handles missing values with tiered strategy
- Engineers features (host experience, amenities count)

### PCA Analysis Pipeline  
```bash
python pca_analysis_pipeline.py
```
- Performs dimensionality reduction
- Identifies 29 principal components explaining 95.47% variance
- Reveals latent patterns in listing characteristics

### Modular Analysis
Each analysis script can run independently using the pre-processed data in `Dataset Processed/`.

## Outputs

- **Statistical models** with coefficient interpretations
- **Visualizations** in `price_analysis_pipeline/results/` folder
- **Processed datasets** with engineered features

## Technical Notes

- Built with Python, pandas, scikit-learn, statsmodels
- Robust statistical controls for property characteristics
- Handles 45,421 LA Airbnb listings
- Modular, reproducible analysis pipelines

---

*Project demonstrating systematic pricing patterns in the LA Airbnb market through rigorous statistical analysis.*