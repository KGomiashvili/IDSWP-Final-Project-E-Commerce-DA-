import pandas as pd
import logging
from config import OCT_DATA_PATH

# Configure Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import pandas as pd
import logging
from config import OCT_DATA_PATH

# Configure Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data():
    """Loads the first nrows from the October dataset."""
    try:
        logger.info(f"Loading all rows from the October dataset...")

        df = pd.read_csv(OCT_DATA_PATH,  skiprows=lambda i: i % 100 != 0)

        logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
        df.to_csv('loaded.csv', index=False)
        logger.info("Saved to 'loaded.csv'.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None



def preprocess_data(df):
    """Preprocesses the e-commerce dataset."""
    try:
        logger.info("Starting preprocessing...")

        # Convert event_time to datetime
        df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')

        # Drop rows with missing crucial values (product_id, category_id, user_id, event_time)
        df.dropna(subset=['product_id', 'category_id', 'user_id', 'event_time'], inplace=True)

        # Fill missing categorical values with 'unknown'
        df['category_code'].fillna('unknown', inplace=True)
        df['brand'].fillna('unknown', inplace=True)

        # Convert event_type to categorical
        df['event_type'] = df['event_type'].astype('category')

        # Convert price to float and handle negative prices (if any)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df[df['price'] >= 0]

        # Sort by event_time
        df.sort_values(by='event_time', inplace=True)

        # Feature Engineering
        df['hour'] = df['event_time'].dt.hour  # Extract hour of event
        df['dayofweek'] = df['event_time'].dt.dayofweek  # Extract day of the week
        
        logger.info(f"Preprocessing complete. Final dataset shape: {df.shape[0]} rows, {df.shape[1]} columns.")
        df.to_csv('cleaned.csv', index=False)
        logger.info("Saved to 'cleaned.csv'.")
        return df
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return None
# # Execute pipeline
# if __name__ == "__main__":
#     df = load_data()
#     if df is not None:
#         df = preprocess_data(df)
#         if df is not None:
#             df.to_csv("cleaned_data.csv", index=False)
#             logger.info("Cleaned data saved successfully.")
