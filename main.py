import argparse
import pandas as pd
from data_analysis import DataAnalysis
from classifcation_helper import CustomerClustering
from recommendation_helper import Recommender
import sys

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run different parts of the customer analysis pipeline.")
    
    # Adding the different arguments
    parser.add_argument("--data_analysis", action="store_true", help="Run the data analysis part")
    parser.add_argument("--plot", action="store_true", help="Run the plot part")
    parser.add_argument("--compute_wcss", action="store_true", help="Compute WCSS for KMeans")
    parser.add_argument("--apply_kmeans", action="store_true", help="Apply KMeans clustering")
    parser.add_argument("--recommendation", action="store_true", help="Run recommendation engine")
    parser.add_argument("--customer_id", type=int, help="Specify customer ID for recommendations")
    parser.add_argument("--all", action="store_true", help="Run all parts of the code with default customer ID 1")
    
    args = parser.parse_args()
    
    # Load the data 
    df = pd.read_csv("purchases.csv") 

    # Default customer_id to 1 unless specified
    customer_id = args.customer_id if args.customer_id else 1
    
    if customer_id < 1 or customer_id > 500:
        print("Invalid customer ID. Please enter a valid customer ID between 1 and 500.")
        sys.exit(1)
    # Data Analysis
    if args.data_analysis or args.all:
        print("\nRunning Data Analysis...")
        analysis = DataAnalysis(df)
        analysis.basic_analysis()

    # Plotting
    if args.plot or args.all:
        analysis = DataAnalysis(df)
        print("\nRunning Plotting...")
        analysis.plot()

    # Compute WCSS
    if args.compute_wcss or args.all:
        print("\nComputing WCSS...")
        clustering = CustomerClustering(df)
        clustering.compute_wcss()

    # Apply KMeans
    if args.apply_kmeans or args.all:
        clustering = CustomerClustering(df)
        print("\nApplying KMeans...")
        clustering.apply_kmeans()
        print("\nPlotting Clusters...")
        clustering.plot_clusters(method="kmeans")

    # Recommendation
    if args.recommendation or args.all:
        print("\nRunning Recommendation Engine...")
        if args.customer_id:
            print(f'Running with Customer ID: {customer_id}')
        else:
            print(f"Running with Customer ID: {customer_id} (assuming customer ID = 1 as no customer ID was provided)")

        # Default to True for both CF and CB when --recommendation or --all is provided
        cf_enabled = True
        cb_enabled = True
    
        # Recommendation logic
        recommendations = []
       
        recommender = Recommender(df, cf_enabled=cf_enabled, cb_enabled=cb_enabled)
        print(f"Top 5 Collaborative and Content Based Recommendations for Customer {customer_id}:")
        recommendations = recommender.hybrid_recommendation(customer_id)
        print(recommendations)

if __name__ == "__main__":
    main()
