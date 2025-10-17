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
   ```bash
   python main.py
