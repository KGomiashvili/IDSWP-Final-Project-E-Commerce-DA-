import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler

# Configure Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def compute_session_duration(df):
    logger.info("Computing session duration...")
    session_times = df.groupby('user_session')['event_time'].agg(['min', 'max'])
    session_times['session_duration'] = (session_times['max'] - session_times['min']).dt.total_seconds()
    return df.merge(session_times[['session_duration']], on='user_session', how='left')

def compute_interaction_counts(df):
    logger.info("Computing interaction counts per session...")
    interaction_counts = df.pivot_table(index='user_session', columns='event_type', values='product_id', aggfunc='count', fill_value=0)
    interaction_counts.columns = [f"session_{col}_count" for col in interaction_counts.columns]
    return df.merge(interaction_counts, on='user_session', how='left')

def compute_product_popularity(df):
    logger.info("Computing product popularity...")
    product_counts = df.pivot_table(index='product_id', columns='event_type', values='user_id', aggfunc='count', fill_value=0)
    product_counts.columns = [f"product_{col}_count" for col in product_counts.columns]
    return df.merge(product_counts, on='product_id', how='left')

def compute_user_activity_patterns(df):
    logger.info("Computing user activity patterns...")
    user_events = df.groupby('user_id').size().reset_index(name='user_event_count')
    return df.merge(user_events, on='user_id', how='left')

def compute_price_sensitivity(df):
    logger.info("Computing price sensitivity...")
    cart_prices = df[df['event_type'] == 'cart'].groupby('product_id')['price'].mean().rename('avg_cart_price')
    purchase_prices = df[df['event_type'] == 'purchase'].groupby('product_id')['price'].mean().rename('avg_purchase_price')

    price_diff = pd.concat([cart_prices, purchase_prices], axis=1)
    price_diff['price_sensitivity'] = price_diff['avg_cart_price'] - price_diff['avg_purchase_price']
    
    return df.merge(price_diff[['price_sensitivity']], on='product_id', how='left')

def preprocess_features(df):
    logger.info("Preprocessing categorical and numerical features...")
    
    df = df.dropna(subset=['category_code', 'brand'])  # Remove missing values

    # Encode categorical variables
    df['category_code'] = df['category_code'].astype('category').cat.codes
    df['brand'] = df['brand'].astype('category').cat.codes

    # Normalize numerical features
    scaler = StandardScaler()
    df[['category_code', 'price', 'brand']] = scaler.fit_transform(df[['category_code', 'price', 'brand']])

    return df

def create_user_item_matrix(df):
    logger.info("Creating user-item interaction matrix...")
    user_item_matrix = df.pivot_table(index='user_id', columns='product_id', values='purchase', fill_value=0)
    user_item_matrix.to_csv('user_item_matrix.csv')  # Save for later use
    return df

def feature_engineering(df):
    logger.info("Starting feature engineering...")

    # Convert event_time to datetime
    df['event_time'] = pd.to_datetime(df['event_time'])

    # Convert event type to purchase flag
    df['purchase'] = (df['event_type'] == 'purchase').astype(int)

    df = compute_session_duration(df)
    df = compute_interaction_counts(df)
    df = compute_product_popularity(df)
    df = compute_user_activity_patterns(df)
    df = compute_price_sensitivity(df)
    df = preprocess_features(df)
    df = create_user_item_matrix(df)

    logger.info("Feature engineering complete.")
    df.to_csv('featured.csv', index=False)
    logger.info("Saved to 'featured.csv'.")
    return df
