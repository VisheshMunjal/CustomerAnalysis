import pandas as pd
import matplotlib.pyplot as plt

class DataAnalysis:
    def __init__(self, df):
        """Initialize with a dataframe."""
        self.df = df
        self.df["purchase_date"] = pd.to_datetime(self.df["purchase_date"])  # Convert dates

    def basic_analysis(self):
        """Performs basic analysis like top-selling products, categories, and avg spending per customer."""
        # Top-selling products
        top_products = self.df.groupby("product_id")["purchase_amount"].sum().sort_values(ascending=False).head(10)
        print("Top 10 selling products' product_id:")
        print(top_products.index.tolist())

        # Top-selling categories
        top_categories = self.df.groupby("product_category")["purchase_amount"].sum().sort_values(ascending=False).head(3)
        print("\nTop-selling categories:")
        print(top_categories.index.tolist())

        # Average spending per customer
        avg_spending_per_customer = self.df.groupby("customer_id")["purchase_amount"].sum().mean()
        print(f"\nAverage spending per customer: ${avg_spending_per_customer:.2f}")

    def plot(self):
        # Aggregate customer data
        customer_data = self.df.groupby("customer_id").agg(
            total_spent=("purchase_amount", "sum"),
            purchase_frequency=("purchase_date", "count"),
            unique_categories=("product_category", "nunique")
        ).reset_index()

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(customer_data["total_spent"], customer_data["purchase_frequency"], 
                   customer_data["unique_categories"], c='blue', label="Customer Data", s=50, alpha=0.7)

        ax.set_xlabel("Total Spent")
        ax.set_ylabel("Purchase Frequency")
        ax.set_zlabel("Unique Categories")
        ax.set_title("Customer Data Exploration (3D View)")

        plt.show()