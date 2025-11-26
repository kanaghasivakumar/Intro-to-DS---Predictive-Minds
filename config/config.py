DATA_PATH = r'C:\NU MSAI Fall 2025\Intro to DS Dataset\listings.csv'
OUTPUT_PATH = 'Dataset Processed/la_airbnb_processed_with_pca.csv'
PCA_COMPONENTS_PATH = 'Dataset Processed/la_airbnb_pca_components.csv'
OUTPUT_PATH2 = 'Dataset Processed/la_airbnb_cleaned.csv'
OUTPUT_PATH3 = 'Dataset Processed/la_airbnb_cleaned_and_missing_values_handled.csv'
N_COMPONENTS = None 

PRICE_COLUMNS = ['price', 'weekly_price', 'monthly_price', 'security_deposit', 'cleaning_fee', 'extra_people']
DATE_COLUMNS = ['last_scraped', 'host_since', 'calendar_last_scraped']
BINARY_FEATURES = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable']