import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

class Recommender:
    def __init__(self, df, cf_enabled=True, cb_enabled=True):
        self.df = df
        self.cf_enabled = cf_enabled
        self.cb_enabled = cb_enabled
        # Create the utility matrix for collaborative filtering
        self.user_product_matrix = df.pivot_table(index='customer_id', columns='product_id', values='purchase_amount', aggfunc='sum', fill_value=0)
        self.user_product_matrix_normalized = self.user_product_matrix.sub(self.user_product_matrix.mean(axis=1), axis=0)
        
        # Content-Based Filtering (CB) preparation
        self.le = LabelEncoder()
        self.df['product_category_encoded'] = self.le.fit_transform(df['product_category'])
        
    def collaborative_filtering(self, customer_id, num_similar_customers=10):
        """Generate recommendations using collaborative filtering."""
        user_similarity = cosine_similarity(self.user_product_matrix_normalized)
        
        # Get similar customers to the target customer
        similar_customers = user_similarity[customer_id - 1]  
        similar_customers_sorted_idx = np.argsort(similar_customers)[::-1]
        
        # Exclude the target customer
        similar_customers_sorted_idx = similar_customers_sorted_idx[similar_customers_sorted_idx != (customer_id - 1)]
        
        # Select top N similar customers
        similar_customers_sorted_idx = similar_customers_sorted_idx[:num_similar_customers]
        
        # Get products purchased by the most similar customers
        recommended_by_cf = []
        for idx in similar_customers_sorted_idx:
            customer_purchases = self.user_product_matrix.iloc[idx]
            products_purchased = customer_purchases[customer_purchases > 0].index.tolist()
            recommended_by_cf.extend(products_purchased)
        
        return list(set(recommended_by_cf))
    
    def content_based_filtering(self, customer_id):
        """Generate recommendations using content-based filtering."""
        customer_categories = self.df[self.df['customer_id'] == customer_id]['product_category_encoded'].unique()
        
        recommended_by_cb = []
        for category in customer_categories:
            products_in_category = self.df[self.df['product_category_encoded'] == category]['product_id'].unique()
            recommended_by_cb.extend(products_in_category)
        
        return list(set(recommended_by_cb))
    
    def hybrid_recommendation(self, customer_id, cf_weight=0.7, cb_weight=0.3, top_n=5, num_similar_customers=10):
        """Generate hybrid recommendations by combining CF and CB results."""
        recommended_products = []
        
        if self.cf_enabled:
            cf_recommendations = self.collaborative_filtering(customer_id, num_similar_customers)
            recommended_products.extend(cf_recommendations)
        
        if self.cb_enabled:
            cb_recommendations = self.content_based_filtering(customer_id)
            recommended_products.extend(cb_recommendations)
        
        # Filter out products the customer has already bought
        products_already_bought = self.user_product_matrix.columns[self.user_product_matrix.loc[customer_id] > 0].tolist()
        recommended_products = [product for product in recommended_products if product not in products_already_bought]
        
        product_scores = {}
        
        # Get user similarity once to use for CF scoring
        user_similarity = cosine_similarity(self.user_product_matrix_normalized)
        similar_customers = user_similarity[customer_id - 1]  
        similar_customers_sorted_idx = np.argsort(similar_customers)[::-1]
        similar_customers_sorted_idx = similar_customers_sorted_idx[similar_customers_sorted_idx != (customer_id - 1)]
        similar_customers_sorted_idx = similar_customers_sorted_idx[:num_similar_customers]
        customer_categories = self.df[self.df['customer_id'] == customer_id]['product_category_encoded'].unique()  # For CB
        
        for product in recommended_products:
            cf_score = 0
            if self.cf_enabled:
                # Collaborative Filtering Score (CF): Based on similarity with similar customers
                for idx in similar_customers_sorted_idx:
                    if product in self.user_product_matrix.columns:
                        cf_score += self.user_product_matrix.iloc[idx][product]
            
            cb_score = 0
            if self.cb_enabled:
                # Content-Based Filtering Score (CB): Based on product category
                for category in customer_categories:
                    if product in self.df[self.df['product_category_encoded'] == category]['product_id'].values:
                        cb_score += 1
            
            combined_score = cf_weight * cf_score + cb_weight * cb_score
            product_scores[product] = combined_score
        
        # Sort by the combined score and return top N recommendations
        sorted_recommendations = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)
        top_recommendations = [x[0] for x in sorted_recommendations[:top_n]]
        
        return top_recommendations