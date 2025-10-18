# run_visualizations.py
import pandas as pd
from visualizations import AirbnbVisualizer
import numpy as np

def load_processed_data():
    """Load the processed data from the PCA pipeline"""
    try:
        # Load the processed data
        cleaned_df = pd.read_csv('la_airbnb_processed_with_pca.csv')
        
        # For components_df, we'll recreate it from the PCA results
        # You can modify this based on how you saved your components
        pca_components = pd.read_csv('la_airbnb_pca_components.csv')
        
        print("✅ Processed data loaded successfully!")
        print(f"Dataset shape: {cleaned_df.shape}")
        
        return cleaned_df, pca_components
        
    except FileNotFoundError:
        print("❌ Processed data files not found!")
        print("💡 Please run main.py first to generate the PCA results")
        return None, None

def main():
    """Main function to run all visualizations"""
    print("🎨 LA Airbnb Data Visualization Dashboard")
    print("=" * 50)
    
    # Load data
    cleaned_df, pca_components = load_processed_data()
    if cleaned_df is None:
        return
    
    # Create sample components_df for visualization (you should replace this with your actual components)
    # This is a placeholder - you'll need to adapt this based on your actual PCA results
    pc_columns = [col for col in cleaned_df.columns if col.startswith('PC')]
    n_components = len(pc_columns)
    
    # Create dummy components_df (replace with your actual component loadings)
    components_df = pd.DataFrame(
        index=[f'PC{i+1}' for i in range(n_components)],
        columns=cleaned_df.select_dtypes(include=[np.number]).columns[:20]  # Sample features
    ).fillna(0)
    
    # Sample variance explained (replace with your actual values)
    explained_variance = [0.1374, 0.0998, 0.0807, 0.0539, 0.0454] + [0.03] * (n_components - 5)
    cumulative_variance = np.cumsum(explained_variance)
    
    # Create visualizer
    visualizer = AirbnbVisualizer(cleaned_df, components_df, explained_variance, cumulative_variance)
    
    # Run all visualizations
    visualizer.run_all_visualizations()

if __name__ == "__main__":
    main()