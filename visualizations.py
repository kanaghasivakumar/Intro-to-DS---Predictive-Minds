# visualizations.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

class AirbnbVisualizer:
    def __init__(self, cleaned_df, components_df, explained_variance, cumulative_variance):
        self.df = cleaned_df
        self.components_df = components_df
        self.explained_variance = explained_variance
        self.cumulative_variance = cumulative_variance
        self.pc_columns = [col for col in self.df.columns if col.startswith('PC')]
        
    def create_pca_summary_plots(self):
        print("Creating PCA Summary Plots")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        components_range = range(1, len(self.explained_variance) + 1)
        axes[0,0].plot(components_range, self.explained_variance, 'bo-', linewidth=2, markersize=6, label='Individual')
        axes[0,0].plot(components_range, self.cumulative_variance, 'ro-', linewidth=2, markersize=6, label='Cumulative')
        axes[0,0].set_xlabel('Principal Components')
        axes[0,0].set_ylabel('Variance Explained')
        axes[0,0].set_title('Scree Plot - PCA Variance Explained', fontsize=14, fontweight='bold')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].axhline(y=0.95, color='g', linestyle='--', alpha=0.7, label='95% Variance')
        
        explained_95 = np.argmax(self.cumulative_variance >= 0.95) + 1
        colors = ['skyblue' if i < explained_95 else 'lightgray' for i in components_range]
        axes[0,1].bar(components_range, self.cumulative_variance, color=colors, alpha=0.8)
        axes[0,1].axhline(y=0.95, color='red', linestyle='--', linewidth=2, label=f'95% Variance (PC{explained_95})')
        axes[0,1].set_xlabel('Number of Components')
        axes[0,1].set_ylabel('Cumulative Variance Explained')
        axes[0,1].set_title('Cumulative Variance by Number of Components', fontsize=14, fontweight='bold')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        top_10_variance = self.explained_variance[:10]
        colors_10 = plt.cm.viridis(np.linspace(0, 1, 10))
        bars = axes[1,0].bar(range(1, 11), top_10_variance, color=colors_10, alpha=0.8)
        axes[1,0].set_xlabel('Principal Components')
        axes[1,0].set_ylabel('Variance Explained')
        axes[1,0].set_title('Top 10 Principal Components - Individual Variance', fontsize=14, fontweight='bold')
        
        for bar, value in zip(bars, top_10_variance):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                          f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        pc1_loadings = self.components_df.iloc[0].sort_values(ascending=False)
        top_10_pc1 = pd.concat([pc1_loadings.head(5), pc1_loadings.tail(5)])
        
        colors_pc1 = ['green' if x > 0 else 'red' for x in top_10_pc1.values]
        bars_pc1 = axes[1,1].barh(range(len(top_10_pc1)), top_10_pc1.values, color=colors_pc1, alpha=0.7)
        axes[1,1].set_yticks(range(len(top_10_pc1)))
        axes[1,1].set_yticklabels(top_10_pc1.index, fontsize=10)
        axes[1,1].set_xlabel('Loading Value')
        axes[1,1].set_title('Top 10 Feature Loadings - PC1 (Review Quality)', fontsize=14, fontweight='bold')
        
        for i, (bar, value) in enumerate(zip(bars_pc1, top_10_pc1.values)):
            axes[1,1].text(value + (0.01 if value > 0 else -0.03), i, f'{value:.3f}', 
                          va='center', fontweight='bold', color='black' if abs(value) > 0.1 else 'gray')
        
        plt.tight_layout()
        plt.savefig('pca_summary_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_pca_scatter_plots(self):
        print("Creating PCA Scatter Plots")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        room_types = self.df['room_type'].value_counts().index
        colors_room = sns.color_palette("Set2", len(room_types))
        
        for i, room_type in enumerate(room_types):
            mask = self.df['room_type'] == room_type
            axes[0,0].scatter(self.df.loc[mask, 'PC1'], self.df.loc[mask, 'PC2'], 
                             alpha=0.6, label=room_type, color=colors_room[i], s=30)
        
        axes[0,0].set_xlabel('PC1 - Review Quality (13.74%)')
        axes[0,0].set_ylabel('PC2 - Property Size (9.98%)')
        axes[0,0].set_title('Review Quality vs Property Size (Colored by Room Type)', fontsize=14, fontweight='bold')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        price_q = pd.qcut(self.df['price'], 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        colors_price = ['green', 'yellow', 'orange', 'red']
        
        for i, (q, color) in enumerate(zip(price_q.unique(), colors_price)):
            mask = price_q == q
            axes[0,1].scatter(self.df.loc[mask, 'PC1'], self.df.loc[mask, 'PC3'], 
                             alpha=0.6, label=f'Price: {q}', color=color, s=30)
        
        axes[0,1].set_xlabel('PC1 - Review Quality (13.74%)')
        axes[0,1].set_ylabel('PC3 - Availability (8.07%)')
        axes[0,1].set_title('Review Quality vs Availability (Colored by Price Quartile)', fontsize=14, fontweight='bold')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        for superhost in [0, 1]:
            mask = self.df['host_is_superhost'] == superhost
            label = 'Superhost' if superhost == 1 else 'Regular Host'
            color = 'gold' if superhost == 1 else 'blue'
            axes[1,0].scatter(self.df.loc[mask, 'PC2'], self.df.loc[mask, 'PC4'], 
                             alpha=0.6, label=label, color=color, s=30)
        
        axes[1,0].set_xlabel('PC2 - Property Size (9.98%)')
        axes[1,0].set_ylabel('PC4 - Popularity (5.39%)')
        axes[1,0].set_title('Property Size vs Popularity (Colored by Superhost Status)', fontsize=14, fontweight='bold')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        scatter = axes[1,1].scatter(self.df['PC1'], self.df['price'], 
                                   c=self.df['review_scores_rating'], 
                                   cmap='viridis', alpha=0.6, s=30)
        axes[1,1].set_xlabel('PC1 - Review Quality (13.74%)')
        axes[1,1].set_ylabel('Price ($)')
        axes[1,1].set_title('Review Quality vs Price (Colored by Rating)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=axes[1,1], label='Review Score Rating')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pca_scatter_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_component_heatmap(self, n_components=10):
        print("Creating Component Loading Heatmap")
        
        top_components = self.components_df.iloc[:n_components]
        
        top_features = set()
        for i in range(n_components):
            component_loadings = top_components.iloc[i].abs().sort_values(ascending=False)
            top_features.update(component_loadings.head(6).index)
        
        heatmap_data = top_components[list(top_features)]
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                   cbar_kws={'label': 'Loading Value'}, linewidths=0.5)
        plt.title(f'Top {n_components} PCA Components - Feature Loadings\n(Red: Positive Correlation, Blue: Negative Correlation)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Features')
        plt.ylabel('Principal Components')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('pca_component_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_neighborhood_analysis(self):
        print("Creating Neighborhood Analysis")
        
        top_neighborhoods = self.df['neighbourhood_cleansed'].value_counts().head(8).index
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        for neighborhood in top_neighborhoods:
            mask = self.df['neighbourhood_cleansed'] == neighborhood
            axes[0,0].scatter(self.df.loc[mask, 'PC1'], self.df.loc[mask, 'PC2'], 
                             alpha=0.7, label=neighborhood, s=40)
        
        axes[0,0].set_xlabel('PC1 - Review Quality')
        axes[0,0].set_ylabel('PC2 - Property Size')
        axes[0,0].set_title('Neighborhood Distribution in PCA Space', fontsize=14, fontweight='bold')
        axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,0].grid(True, alpha=0.3)
        
        neighborhood_pc_means = self.df.groupby('neighbourhood_cleansed')[['PC1', 'PC2', 'PC3', 'PC4']].mean()
        neighborhood_pc_means = neighborhood_pc_means.loc[top_neighborhoods]
        
        neighborhood_pc_means.plot(kind='bar', ax=axes[0,1], figsize=(12, 6))
        axes[0,1].set_title('Average Principal Component Values by Neighborhood', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('Average PC Value')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        for neighborhood in top_neighborhoods[:4]:
            mask = self.df['neighbourhood_cleansed'] == neighborhood
            axes[1,0].scatter(self.df.loc[mask, 'PC1'], self.df.loc[mask, 'price'], 
                             alpha=0.6, label=neighborhood, s=40)
        
        axes[1,0].set_xlabel('PC1 - Review Quality')
        axes[1,0].set_ylabel('Price ($)')
        axes[1,0].set_title('Price vs Review Quality by Neighborhood', fontsize=14, fontweight='bold')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        superhost_ratio = self.df.groupby('neighbourhood_cleansed')['host_is_superhost'].mean().sort_values(ascending=False).head(10)
        superhost_ratio.plot(kind='bar', ax=axes[1,1], color='gold', alpha=0.7)
        axes[1,1].set_title('Superhost Ratio by Neighborhood (Top 10)', fontsize=14, fontweight='bold')
        axes[1,1].set_ylabel('Superhost Ratio')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('neighborhood_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_price_analysis(self):
        print("Creating Price Analysis")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        self.df.boxplot(column='price', by='room_type', ax=axes[0,0])
        axes[0,0].set_title('Price Distribution by Room Type', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('Price ($)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        top_pcs = self.pc_columns[:8]
        price_correlations = [self.df[pc].corr(self.df['price']) for pc in top_pcs]
        
        bars = axes[0,1].bar(top_pcs, price_correlations, 
                            color=['red' if x < 0 else 'green' for x in price_correlations],
                            alpha=0.7)
        axes[0,1].set_title('Correlation: Price vs Principal Components', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('Correlation Coefficient')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, price_correlations):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.01 if value > 0 else -0.02), 
                          f'{value:.3f}', ha='center', va='bottom' if value > 0 else 'top', fontweight='bold')
        
        scatter = axes[1,0].scatter(self.df['PC2'], self.df['price'], 
                                   c=self.df['accommodates'], cmap='plasma', alpha=0.6, s=30)
        axes[1,0].set_xlabel('PC2 - Property Size')
        axes[1,0].set_ylabel('Price ($)')
        axes[1,0].set_title('Property Size vs Price (Colored by Accommodates)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=axes[1,0], label='Accommodates')
        axes[1,0].grid(True, alpha=0.3)
        
        scatter = axes[1,1].scatter(self.df['number_of_reviews'], self.df['price'], 
                                   s=self.df['PC1'].abs() * 50,  
                                   c=self.df['review_scores_rating'], cmap='viridis', alpha=0.6)
        axes[1,1].set_xlabel('Number of Reviews')
        axes[1,1].set_ylabel('Price ($)')
        axes[1,1].set_title('Price vs Reviews (Size: Review Quality, Color: Rating)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=axes[1,1], label='Review Score Rating')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('price_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_interactive_3d_plot(self):
        try:
            print("Creating Interactive 3D Plot")
            
            fig = px.scatter_3d(
                self.df.head(2000), 
                x='PC1',
                y='PC2', 
                z='PC3',
                color='room_type',
                size='price',
                hover_data=['neighbourhood_cleansed', 'review_scores_rating', 'accommodates'],
                title='3D PCA Space: Review Quality vs Property Size vs Availability',
                labels={
                    'PC1': 'Review Quality (13.74%)',
                    'PC2': 'Property Size (9.98%)', 
                    'PC3': 'Availability (8.07%)'
                }
            )
            
            fig.write_html("pca_3d_interactive.html")
            print("Interactive 3D plot saved as 'pca_3d_interactive.html'")
            
        except ImportError:
            print("Plotly not available for interactive 3D plots")
            
    def run_all_visualizations(self):
        print("Generating All Visualizations\n")
        
        self.create_pca_summary_plots()
        self.create_pca_scatter_plots() 
        self.create_component_heatmap(n_components=10)
        self.create_neighborhood_analysis()
        self.create_price_analysis()
        self.create_interactive_3d_plot()
        
        print("\nAll visualizations completed!")
        print("Files created:")
        print("   - pca_summary_plots.png")
        print("   - pca_scatter_plots.png") 
        print("   - pca_component_heatmap.png")
        print("   - neighborhood_analysis.png")
        print("   - price_analysis.png")
        print("   - pca_3d_interactive.html (if plotly available)")