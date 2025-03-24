# BookedBy Task

This task is a customer analysis one that processes and analyzes customer purchase data, applies clustering algorithms, and generates product recommendations based on the data. The project is designed to allow easy command-line execution of the tasks through various arguments.

## Files Overview

### `generate_data.py`
This script is responsible for generating the `purchases.csv` file, which contains simulated customer purchase data. The generated file is used as input for all the tasks in the analysis pipeline. You can run this script to generate fresh purchase data if needed(**Note**: regeneration of data would mean that clusters might need to be re-labeled).

### `purchases.csv`
This file contains customer purchase data in CSV format. Each row represents a single purchase made by a customer, and the columns contain details such as customer ID, product ID, quantity, and purchase date.

### `data_analysis.py`
This module contains the `DataAnalysis` class, which performs various data analysis tasks on the customer purchase data. Key tasks include:
- Basic data analysis 
- Visualizations for data exploration

### `classification_helper.py`
This module contains the `CustomerClustering` class, which is used to apply clustering algorithms to the customer purchase data. Specifically, it:
- Computes WCSS (Within-Cluster Sum of Squares) for KMeans clustering.
- Applies KMeans clustering to the data.
- Generates and visualizes customer clusters.
- Outputs a CSV file (`customer_segments_kmeans.csv`) that assigns each customer to a specific cluster.

### `recommendation_helper.py`
This module contains the `Recommender` class, which provides recommendation functionalities for customers based on their purchasing behaviors. It supports:
- Collaborative Filtering (CF) recommendations.
- Content-Based (CB) recommendations.
- Hybrid recommendations combining both CF and CB approaches.

### `main.py`
This is the main script of the pipeline. It can be run from the command line with different arguments to execute the tasks. The available tasks include:
- **Data Analysis**: Basic analysis of the data.
- **Plotting**: Visualizing the data.
- **Clustering**: Applying KMeans clustering to segment customers.
- **Recommendation Engine**: Generating recommendations for a specified customer.

### `requirements.txt`
Has the dependencies required to run the code.

## Installing Dependecies

Run the command
```bash
pip install -r requirements.txt
```

## How to Run the Pipeline

To run the pipeline, execute the following command from the command line:

```bash
python main.py [options]
```
### Available options

- --data_analysis: Run the data analysis tasks (basic analysis).

- --plot: Plots a representation of feature(total amount spend, frequency of purchases, category of purchases) for all customers (visualizations).

- --compute_wcss: Compute the WCSS for KMeans.

- --apply_kmeans: Apply KMeans clustering and plot the clusters.

- --recommendation: Run the recommendation engine (with optional customer ID).

- --customer_id <id>: Specify the customer ID for the recommendation engine. The default ID is 1 if not specified.

- --all: Run all tasks in the pipeline, using the default customer ID 1.

### Examples

- Run all task (**Note**: Based on the place of running, plots might appear one at a time, and the other might only come after the first one is closed.)
```bash
python main.py --all
```

- Run only KMeans clustering:
```bash
python main.py --apply_kmeans
```

- Run only recommendation for customer ID 7
```bash
python main.py --recommendation --customer_id 7
```

