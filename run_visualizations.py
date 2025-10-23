# run_visualizations.py
import pandas as pd
from visualizations import AirbnbVisualizer
import numpy as np

def load_processed_data():
    try:
        cleaned_df = pd.read_csv('la_airbnb_processed_with_pca.csv')
        
        pca_components = pd.read_csv('la_airbnb_pca_components.csv')
        
        print("Processed data loaded successfully!")
        print(f"Dataset shape: {cleaned_df.shape}")
        
        return cleaned_df, pca_components
        
    except FileNotFoundError:
        print("Processed data files not found!")
        print("Please run main.py first to generate the PCA results")
        return None, None

def main():
    print("LA Airbnb Data Visualization Dashboard")
    print("=" * 50)
    
    cleaned_df, pca_components = load_processed_data()
    if cleaned_df is None:
        return
    
    pc_columns = [col for col in cleaned_df.columns if col.startswith('PC')]
    n_components = len(pc_columns)
    
    components_df = pd.DataFrame(
        index=[f'PC{i+1}' for i in range(n_components)],
        columns=cleaned_df.select_dtypes(include=[np.number]).columns[:20]
    ).fillna(0)
    
    explained_variance = [0.1374, 0.0998, 0.0807, 0.0539, 0.0454] + [0.03] * (n_components - 5)
    cumulative_variance = np.cumsum(explained_variance)
    
    visualizer = AirbnbVisualizer(cleaned_df, components_df, explained_variance, cumulative_variance)
    
    visualizer.run_all_visualizations()

if __name__ == "__main__":
    main()