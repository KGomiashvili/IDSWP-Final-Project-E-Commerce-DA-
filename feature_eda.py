import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_processed_data(file_path="processed_with_features.csv"):
    """Loads the preprocessed dataset with engineered features."""
    try:
        logger.info(f"Loading processed data from {file_path}...")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        return None

def summary_statistics(df):
    """Displays summary statistics for numerical features."""
    logger.info("Generating summary statistics...")
    print(df.describe())

def visualize_distribution(df, column, bins=30):
    """Plots histogram and boxplot for a given numerical column."""
    plt.figure(figsize=(12, 5))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(df[column].dropna(), bins=bins, kde=True)
    plt.title(f"Distribution of {column}")

    # Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[column].dropna())
    plt.title(f"Boxplot of {column}")

    plt.show()

def correlation_matrix(df):
    """Plots correlation heatmap for numerical features."""
    logger.info("Generating correlation matrix...")

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])

    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.show()


def analyze_features(df):
    """Performs EDA on engineered features."""
    logger.info("Starting EDA on engineered features...")

    summary_statistics(df)

    # Visualizing each engineered feature
    engineered_features = [
        'session_duration', 'session_view_count', 'session_cart_count', 'session_purchase_count',
        'product_view_count', 'product_purchase_count', 'user_event_count', 'price_sensitivity'
    ]
    
    for feature in engineered_features:
        if feature in df.columns:
            visualize_distribution(df, feature)

    correlation_matrix(df)
    logger.info("EDA complete.")

# if __name__ == "__main__":
#     df = load_processed_data()
#     if df is not None:
#         analyze_features(df)
