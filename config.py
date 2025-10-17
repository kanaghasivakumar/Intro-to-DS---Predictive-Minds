# Configuration constants
DATA_PATH = r'C:\NU MSAI Fall 2025\Intro to DS Dataset\listings.csv'
OUTPUT_PATH = 'la_airbnb_processed_with_pca.csv'
PCA_COMPONENTS_PATH = 'la_airbnb_pca_components.csv'

# PCA settings
N_COMPONENTS = None  # Set to specific number or None for automatic selection

# Feature lists
PRICE_COLUMNS = ['price', 'weekly_price', 'monthly_price', 'security_deposit', 'cleaning_fee', 'extra_people']
DATE_COLUMNS = ['last_scraped', 'host_since', 'calendar_last_scraped']
BINARY_FEATURES = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable']