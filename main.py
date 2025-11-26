# main.py (optional orchestrator)
from data_cleaning_pipeline import run_data_cleaning_pipeline
from pca_analysis_pipeline import run_pca_analysis_pipeline

def main():
    print("LA Airbnb Analysis - Choose Pipeline:")
    print("1. Data Cleaning Pipeline")
    print("2. PCA Analysis Pipeline")
    print("3. Both Pipelines")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        run_data_cleaning_pipeline()
    elif choice == "2":
        run_pca_analysis_pipeline()
    elif choice == "3":
        print("\nRunning Data Cleaning Pipeline...")
        cleaning_results = run_data_cleaning_pipeline()
        print("\nRunning PCA Analysis Pipeline...")
        pca_results = run_pca_analysis_pipeline()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()