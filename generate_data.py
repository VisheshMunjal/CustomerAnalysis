import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(0)  

# Generate customer IDs and product IDs
customer_ids = np.random.choice(range(1, 501), size=5000, replace=True)
product_ids = np.random.choice(range(1, 51), size=5000, replace=True)

unique_customer_ids = np.random.choice(range(1, 501), size=500, replace=False)
unique_product_ids = np.random.choice(range(1, 51), size=50, replace=False)

customer_ids[:500] = unique_customer_ids
product_ids[:50] = unique_product_ids

purchases = [] #populating the purchase records
for i in range(5000):
    customer_id = customer_ids[i]
    product_id = product_ids[i]
    product_category = np.random.choice(['Electronics', 'Fashion', 'Home Goods', 'Stationary', 'Confectionary', 'Groceries'])
    purchase_amount = round(np.random.uniform(10.0, 1000.0), 2)
    purchase_date = (datetime.now() - timedelta(days=np.random.randint(0, 365))).strftime('%Y-%m-%d')
    purchases.append({
        'customer_id': customer_id,
        'product_id': product_id,
        'product_category': product_category,
        'purchase_amount': purchase_amount,
        'purchase_date': purchase_date
    })

purchases_df = pd.DataFrame(purchases)
purchases_df.to_csv('purchases.csv', index=False)

print(purchases_df.head())
