import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class CustomerClustering:
    def __init__(self, df):
        self.df = df
        self.df["purchase_date"] = pd.to_datetime(self.df["purchase_date"])
        self.customer_data = self._prepare_data()
        self.scaler = StandardScaler()
        self.customer_data_scaled = self.scaler.fit_transform(self.customer_data[["total_spent", "purchase_frequency", "unique_categories"]])

    def _prepare_data(self): # aggregation of data
        return self.df.groupby("customer_id").agg(
            total_spent=("purchase_amount", "sum"),
            purchase_frequency=("purchase_date", "count"),
            unique_categories=("product_category", "nunique")
        ).reset_index()

    def compute_wcss(self, max_k=10):
        wcss = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.customer_data_scaled)
            wcss.append(kmeans.inertia_)

        # Plot Elbow Method
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, max_k + 1), wcss, marker="o", linestyle="--")
        plt.xlabel("Number of Clusters")
        plt.ylabel("WCSS")
        plt.title("Elbow Method for Optimal k")
        plt.show()

    def apply_kmeans(self, k=3, weight_factors=[1.5, 0.8, 1]):
        customer_data_scaled_weighted = self.customer_data_scaled * weight_factors
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.customer_data["cluster"] = kmeans.fit_predict(customer_data_scaled_weighted)

        cluster_labels = {
            0: "Frequent Shoppers and Medium Spenders",
            1: "High Spenders",
            2: "Low Spenders and Less Frequent Shoppers"
        }
        self.customer_data["segment"] = self.customer_data["cluster"].map(cluster_labels)

        self.customer_data.to_csv("customer_segments_kmeans.csv", index=False)
        print("K-Means clustering completed! Results saved to 'customer_segments_kmeans.csv'.")


    def plot_clusters(self, method="kmeans"):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        if method == "kmeans":
            colors = {"High Spenders": "red", "Frequent Shoppers and Medium Spenders": "blue", "Low Spenders and Less Frequent Shoppers": "green"}

        for segment, color in colors.items():
            subset = self.customer_data[self.customer_data["segment"] == segment]
            ax.scatter(subset["total_spent"], subset["purchase_frequency"], subset["unique_categories"],
                       c=[color], label=segment, s=50, alpha=0.7)

        ax.set_xlabel("Total Spent")
        ax.set_ylabel("Purchase Frequency")
        ax.set_zlabel("Unique Categories")
        ax.set_title(f"Customer Segmentation ({method.upper()})")
        ax.legend()
        plt.show()
