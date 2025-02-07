from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
def segment_customers(df, num_clusters=3):
    logger.info("Segmenting customers using K-Means clustering based on their purchasing behavior.")
    """
    Segments customers using K-Means clustering based on their purchasing behavior.
    """
    features = ['session_cart_count', 'session_purchase_count', 'session_view_count',
                'user_event_count', 'price_sensitivity']

    df_cluster = df[features].copy()

    # Handle missing values
    df_cluster.fillna(df_cluster.median(), inplace=True)

    scaler = StandardScaler()
    df_cluster_scaled = scaler.fit_transform(df_cluster)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['customer_segment'] = kmeans.fit_predict(df_cluster_scaled)
    logger.info("Finished Sccessfully")
    
    return df, kmeans, scaler

def plot_elbow_method(df):
    logger.info("Determining the optimal number of clusters using the Elbow Method.")
    """
    Determines the optimal number of clusters using the Elbow Method.
    """
    features = ['session_cart_count', 'session_purchase_count', 'session_view_count',
                'user_event_count', 'price_sensitivity']
    
    df_cluster = df[features].copy()

    # Handle missing values
    df_cluster.fillna(df_cluster.median(), inplace=True)  # Replace NaNs with median

    scaler = StandardScaler()
    df_cluster_scaled = scaler.fit_transform(df_cluster)

    distortions = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_cluster_scaled)
        distortions.append(kmeans.inertia_)

    # Plot elbow method
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, distortions, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
    plt.title('Elbow Method for Optimal K')
    plt.show()

def analyze_segments(df):
    logger.info("Analyzing customer segments with visualization")
    """
    Analyzes customer segments with visualization.

    Parameters:
        df (pd.DataFrame): The dataset containing customer segments.
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='customer_segment', y='session_purchase_count', data=df)
    plt.xlabel('Customer Segment')
    plt.ylabel('Purchase Count')
    plt.title('Purchase Count Distribution Across Customer Segments')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='customer_segment', y='price_sensitivity', data=df)
    plt.xlabel('Customer Segment')
    plt.ylabel('Price Sensitivity')
    plt.title('Price Sensitivity Across Customer Segments')
    plt.show()


import seaborn as sns

def visualize_segment_distribution(df, features):
    logger.info("Visualizing the distribution of customer segments across key features.")
    """
    Visualizes the distribution of customer segments across key features.
    """
    for feature in features:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x='customer_segment', y=feature)
        plt.title(f'Distribution of {feature} by Customer Segment')
        plt.xlabel('Customer Segment')
        plt.ylabel(feature)
        plt.show()

from sklearn.decomposition import PCA

def visualize_clusters_2d(df, features):
    logger.info("Visualizing customer segments on a 2D plot using PCA.")
    """
    Visualizes customer segments on a 2D plot using PCA.
    """
    df_cluster = df[features].copy()
    df_cluster.fillna(df_cluster.median(), inplace=True)

    scaler = StandardScaler()
    df_cluster_scaled = scaler.fit_transform(df_cluster)

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df_cluster_scaled)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_components[:, 0], pca_components[:, 1], c=df['customer_segment'], cmap='viridis', s=50, alpha=0.6)
    plt.title(f'Customer Segments Visualized in 2D Space (PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Customer Segment')
    plt.show()

from sklearn.metrics import silhouette_score

def calculate_silhouette_score(df, features, n_clusters):
    logger.info("Calculating the silhouette score for the clustering quality.")
    """
    Calculates the silhouette score for the clustering quality.
    """
    df_cluster = df[features].copy()
    df_cluster.fillna(df_cluster.median(), inplace=True)

    scaler = StandardScaler()
    df_cluster_scaled = scaler.fit_transform(df_cluster)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(df_cluster_scaled)

    score = silhouette_score(df_cluster_scaled, cluster_labels)
    print(f"Silhouette Score for {n_clusters} clusters: {score:.3f}")

def plot_feature_correlation_heatmap(df, features):
    logger.info("Plotting a heatmap of the correlations between selected features.")
    """
    Plots a heatmap of the correlations between selected features.
    """
    df_cluster = df[features].copy()
    df_cluster.fillna(df_cluster.median(), inplace=True)
    
    corr = df_cluster.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title('Feature Correlation Heatmap')
    plt.show()

def visualize_cluster_size(df):
    logger.info("Visualizing the number of customers in each cluster.")
    
    """
    Visualizes the number of customers in each cluster.
    """
    cluster_size = df['customer_segment'].value_counts()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=cluster_size.index, y=cluster_size.values, palette='Set2')
    plt.title('Number of Customers in Each Segment')
    plt.xlabel('Customer Segment')
    plt.ylabel('Number of Customers')
    plt.show()
