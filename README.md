# README: Customer Behavior Analytics and Recommendation System for E-commerce Platforms

## Project Overview

This project aims to analyze customer interactions on an e-commerce platform using transaction data from October 2019. The objectives include:

1. **Data Preprocessing**: Cleaning and structuring raw data.
2. **Exploratory Data Analysis (EDA)**: Understanding customer behavior and identifying trends.
3. **Feature Engineering**: Creating new features to enhance predictive models.
4. **Feature Analysis**: Evaluating the impact of engineered features on customer behavior.
5. **Machine Learning Models**: Predicting purchases and segmenting customers.
6. **Customer Segmentation**: Grouping customers based on their behavior.
7. **Recommendation System**: Providing personalized product recommendations.

## File Structure

```
.
├── data_preprocessing.py        # Loads and cleans the dataset
├── eda.py                       # Performs exploratory data analysis
├── feature_engineering.py       # Extracts additional insights from the data
├── feature_eda.py               # Analyzes the engineered features
├── ml_models.py                 # Purchase prediction models
├── customer_segmentation.py     # Customer segmentation analysis
├── recommendation.py            # Recommendation system (content & collaborative)
├── recommendation_check.py      # Testing and evaluation of recommendations
├── config.py                    # Stores dataset paths
├── requirements.txt             # Lists required Python libraries
├── main.pdf                     # PDF version of main notebook
├── main.html                    # HTML version of main notebook
└── README.md                    # Documentation
```

---

## Dataset Information
Source: https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store

### **Overview**

The dataset contains **customer behavior data from October 2019**, collected from a large multi-category online store. Each row represents a specific event related to **users and products**, forming a **many-to-many relationship** between them. The data was sourced from the **Open CDP project**, an open-source customer data platform.

### **File Structure**

Each row in the dataset consists of the following attributes:

- **`event_time`** – Timestamp of the event (UTC).
- **`event_type`** – Type of event (`view`, `cart`, `remove_from_cart`, `purchase`).
- **`product_id`** – Unique identifier of the product.
- **`category_id`** – Identifier of the product’s category.
- **`category_code`** – Taxonomy code of the product category (if available).
- **`brand`** – Brand name of the product (may be missing).
- **`price`** – Product price as a floating-point number.
- **`user_id`** – Unique identifier of the user.
- **`user_session`** – Temporary session ID that remains the same throughout a user’s browsing session but changes after a long pause.

### **Event Types**

The dataset captures four types of user interactions:

1. **`view`** – A user viewed a product.
2. **`cart`** – A user added a product to their shopping cart.
3. **`remove_from_cart`** – A user removed a product from their shopping cart.
4. **`purchase`** – A user completed a purchase.

### **Additional Notes**

- A **single user session** can contain **multiple purchases**, representing a single order.
- The dataset provides a **valuable resource** for analyzing user behavior, identifying trends, and improving recommendation systems.

## For further exploration, additional datasets covering different time periods and store categories are available on Kaggle.

## **1. Data Preprocessing (`data_preprocessing.py`)**

### **Purpose**

- Loads the dataset and preprocesses it for analysis.

### **Key Functionalities**

1. **`load_data()`**
   - Loads data from `OCT_DATA_PATH`, reading every 10th row to reduce memory usage.
   - Logs the number of rows and columns.
2. **`preprocess_data(df)`**
   - Converts `event_time` to datetime.
   - Drops missing values in crucial columns.
   - Fills missing categorical values (`category_code`, `brand`) with `"unknown"`.
   - Ensures `price` is positive.
   - Sorts data by event time.
   - Adds features like `hour` and `dayofweek`.

---

## **2. Exploratory Data Analysis (`eda.py`)**

### **Purpose**

- Performs EDA to understand user interactions.

### **Key Functionalities**

1. **`plot_purchase_funnel(df)`** - Visualizes the transition from product views → cart additions → purchases.
2. **`plot_time_series_purchases(df)`** - Analyzes daily purchase trends.
3. **`plot_top_categories(df)`** - Shows the top 10 most popular product categories.
4. **`plot_top_brands(df)`** - Displays the top 10 most purchased brands.
5. **`plot_user_activity_distribution(df)`** - Examines the frequency of user purchases.

---

## **3. Feature Engineering (`feature_engineering.py`)**

### **Purpose**

- Creates **new features** to enhance predictive modeling.

### **Key Functionalities**

1. **`compute_session_duration(df)`** - Computes session duration per `user_session`.
2. **`compute_interaction_counts(df)`** - Calculates total views, cart additions, and purchases per session.
3. **`compute_product_popularity(df)`** - Tracks how often a product is viewed and purchased.
4. **`compute_user_activity_patterns(df)`** - Counts the total number of events per user.
5. **`compute_price_sensitivity(df)`** - Measures price difference between carted and purchased products.
6. **`preprocess_features(df)`** - Encodes categorical features (`category_code`, `brand`) and normalizes numerical features using `StandardScaler`.
7. **`create_user_item_matrix(df)`** - Generates a user-item interaction matrix for collaborative filtering.
8. **`feature_engineering(df)`** - Calls all above functions, adds a binary `purchase` column, and saves the transformed dataset as `featured.csv`.

---

## **4. Feature Analysis (`feature_eda.py`)**

### **Purpose**

- Analyzes and visualizes engineered features.

### **Key Functionalities**

1. **`summary_statistics(df)`** - Prints summary statistics of numerical features.
2. **`visualize_distribution(df, column)`** - Displays histograms and boxplots for numerical columns.
3. **`correlation_matrix(df)`** - Generates a heatmap of feature correlations.
4. **`analyze_features(df)`** - Runs all EDA functions on engineered features.

---

## **5. Machine Learning Models (`ml_models.py`)**

### **Purpose**

- Predicts purchases using machine learning models.

### **Features**

- Uses `RandomForestClassifier` and `XGBClassifier`.
- Evaluates models using Accuracy, Precision, Recall, and F1 Score.

### **Usage**

```python
from ml_models import train_and_evaluate
import pandas as pd

df = pd.read_csv("processed_with_features.csv")
model, scaler, metrics = train_and_evaluate(df, model_type='random_forest')
```

---

## **6. Customer Segmentation (`customer_segmentation.py`)**

### **Purpose**

- Segments customers based on their behavior.

### **Features**

- Uses `KMeans` clustering.
- Determines the optimal number of clusters with the elbow method.
- Visualizes segments using PCA.

### **Usage**

```python
from customer_segmentation import segment_customers, plot_elbow_method

df = pd.read_csv("processed_with_features.csv")
df_segmented, kmeans, scaler = segment_customers(df, num_clusters=4)
plot_elbow_method(df)
```

---

## **7. Recommendation System (`recommendation.py`)**

### **Purpose**

- Provides personalized product recommendations.

### **Features**

- **Content-Based Filtering**: Uses product attributes like `category_code`, `brand`, and `price`, leveraging cosine similarity for recommendations.
- **Collaborative Filtering**: Uses a precomputed user-item interaction matrix and K-Nearest Neighbors (KNN) to recommend products.

### **Usage**

```python
from recommendation import content_based_recommendations, collaborative_filtering_recommendations
import pandas as pd

df = pd.read_csv("processed_with_features.csv")

# Example Usage
sample_product_id = df['product_id'].iloc[0]
sample_user_id = df['user_id'].iloc[0]

content_recs = content_based_recommendations(df, sample_product_id, top_n=5)
collab_recs = collaborative_filtering_recommendations(sample_user_id, top_n=5)
```

---

## **8. Recommendation Testing & Evaluation (`recommendation_check.py`)**

### **Purpose**

- Tests and visualizes recommendation effectiveness.

### **Key Functionalities**

1. **`precision_at_k(actual, predicted, k)`** – Computes the proportion of recommended items that were actually purchased.
2. **`recall_at_k(actual, predicted, k)`** – Measures how many of the purchased items were recommended.
3. **`hit_rate(actual, predicted)`** – Checks if any actual purchases appear in recommendations.
4. **`plot_precision_recall_curve(actual_list, predicted_list, k_values)`** – Visualizes Precision vs. Recall for different `k` values.

---

## **Setup Instructions**

### **1. Install Dependencies**

Ensure you have Python installed, then run:

```bash
pip install -r requirements.txt
```

### **2. Run Preprocessing**

```bash
python data_preprocessing.py
```

### **3. Run EDA**

```bash
python eda.py
```

### **4. Perform Feature Engineering**

```bash
python feature_engineering.py
```

### **5. Analyze Engineered Features**

```bash
python feature_eda.py
```

### **6. Train Machine Learning Models**

```bash
python ml_models.py
```

### **7. Perform Customer Segmentation**

```bash
python customer_segmentation.py
```

### **8. Generate Recommendations**

```bash
python recommendation.py
```

### **9. Test Recommendations**

```bash
python recommendation_check.py
```

---

## **Summary**

This project follows a structured pipeline to analyze customer behavior and improve e-commerce platforms through:

1. **Data Cleaning and Preprocessing**
2. **Exploratory Data Analysis**
3. **Feature Engineering**
4. **Machine Learning for Purchase Prediction**
5. **Customer Segmentation**
6. **Personalized Product Recommendations**

Future steps will involve **improving model performance** and **deploying the system for real-time recommendations**.

**Author**: Konstantine Gomiashvili
