import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from logger import get_logger

# Initialize Logger
logger = get_logger(__name__)

def plot_purchase_funnel(df):
    """Plots the conversion rates from views to purchases."""
    logger.info("Generating purchase funnel visualization...")
    event_counts = df["event_type"].value_counts()
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=event_counts.index, y=event_counts.values, palette="viridis")
    plt.xlabel("Event Type")
    plt.ylabel("Count")
    plt.title("Purchase Funnel: Views → Cart → Purchase")
    plt.show()

def plot_time_series_purchases(df):
    """Plots purchase trends over time."""
    logger.info("Generating time series analysis of purchases...")
    df["event_time"] = pd.to_datetime(df["event_time"])
    purchase_data = df[df["event_type"] == "purchase"].copy()
    purchase_data["date"] = purchase_data["event_time"].dt.date
    daily_purchases = purchase_data.groupby("date").size()

    plt.figure(figsize=(12, 5))
    sns.lineplot(x=daily_purchases.index, y=daily_purchases.values, marker="o")
    plt.xlabel("Date")
    plt.ylabel("Number of Purchases")
    plt.title("Purchase Trends Over Time")
    plt.xticks(rotation=45)
    plt.show()

def plot_top_categories(df):
    """Plots the most popular product categories."""
    logger.info("Generating top product categories visualization...")
    top_categories = df["category_code"].value_counts().head(10)

    plt.figure(figsize=(10, 5))
    sns.barplot(y=top_categories.index, x=top_categories.values, palette="coolwarm")
    plt.xlabel("Count")
    plt.ylabel("Category")
    plt.title("Top 10 Product Categories")
    plt.show()

def plot_top_brands(df):
    """Plots the most popular brands."""
    logger.info("Generating top brands visualization...")
    top_brands = df["brand"].value_counts().head(10)

    plt.figure(figsize=(10, 5))
    sns.barplot(y=top_brands.index, x=top_brands.values, palette="magma")
    plt.xlabel("Count")
    plt.ylabel("Brand")
    plt.title("Top 10 Brands")
    plt.show()

def plot_user_activity_distribution(df):
    """Plots the distribution of user purchase frequency."""
    logger.info("Generating user activity distribution visualization...")
    user_purchases = df[df["event_type"] == "purchase"]["user_id"].value_counts()

    plt.figure(figsize=(10, 5))
    sns.histplot(user_purchases, bins=50, kde=True)
    plt.xlabel("Number of Purchases per User")
    plt.ylabel("Frequency")
    plt.title("User Purchase Distribution")
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from logger import get_logger

# Initialize Logger
logger = get_logger(__name__)

def plot_purchase_funnel(df):
    """Plots the conversion rates from views to purchases."""
    logger.info("Generating purchase funnel visualization...")
    
    event_counts = df["event_type"].value_counts()

    plt.figure(figsize=(8, 5))
    sns.barplot(x=event_counts.index, y=event_counts.values, palette="viridis")
    plt.xlabel("Event Type")
    plt.ylabel("Count")
    plt.title("Purchase Funnel: Views → Cart → Purchase")
    plt.show()

def plot_time_series_purchases(df):
    """Plots purchase trends over time."""
    logger.info("Generating time series analysis of purchases...")
    
    df["event_time"] = pd.to_datetime(df["event_time"])
    purchase_data = df[df["event_type"] == "purchase"].copy()
    purchase_data["date"] = purchase_data["event_time"].dt.date
    daily_purchases = purchase_data.groupby("date").size()

    plt.figure(figsize=(12, 5))
    sns.lineplot(x=daily_purchases.index, y=daily_purchases.values, marker="o")
    plt.xlabel("Date")
    plt.ylabel("Number of Purchases")
    plt.title("Purchase Trends Over Time")
    plt.xticks(rotation=45)
    plt.show()

def plot_top_categories(df):
    """Plots the most popular product categories."""
    logger.info("Generating top product categories visualization...")
    
    top_categories = df["category_code"].value_counts().head(10)

    plt.figure(figsize=(10, 5))
    sns.barplot(y=top_categories.index, x=top_categories.values, palette="coolwarm")
    plt.xlabel("Count")
    plt.ylabel("Category")
    plt.title("Top 10 Product Categories")
    plt.show()

def plot_top_brands(df):
    """Plots the most popular brands."""
    logger.info("Generating top brands visualization...")
    
    top_brands = df["brand"].value_counts().head(10)

    plt.figure(figsize=(10, 5))
    sns.barplot(y=top_brands.index, x=top_brands.values, palette="magma")
    plt.xlabel("Count")
    plt.ylabel("Brand")
    plt.title("Top 10 Brands")
    plt.show()

def plot_user_purchase_frequency(df):
    print("hi")
    """Plots the distribution of purchases per user with X-axis limited to 30 units."""
    logger.info("Generating user purchase frequency visualization...")
    
    user_purchases = df[df["event_type"] == "purchase"]["user_id"].value_counts()

    # Filter values to be within the range 0-30
    user_purchases = user_purchases[user_purchases <= 30]

    plt.figure(figsize=(10, 5))
    sns.histplot(user_purchases, bins=30, kde=True)
    plt.xlabel("Number of Purchases per User")
    plt.ylabel("Frequency")
    plt.title("User Purchase Frequency Distribution")
    plt.xlim(0, 30)  # Explicitly set x-axis limit
    plt.show()


def plot_event_distribution(df):
    """Plots user interactions by event type (view, cart, purchase)."""
    logger.info("Generating user interaction event distribution visualization...")

    plt.figure(figsize=(8, 5))
    sns.histplot(df["event_type"], discrete=True, shrink=0.8, palette="Set2")
    plt.xlabel("Event Type")
    plt.ylabel("Count")
    plt.title("User Interactions by Event Type")
    plt.show()

def perform_eda(df):
    """Runs all EDA functions."""
    try:
        logger.info("Starting Exploratory Data Analysis (EDA)...")
        plot_purchase_funnel(df)
        plot_time_series_purchases(df)
        plot_top_categories(df)
        plot_top_brands(df)
        plot_user_activity_distribution(df)
        logger.info("EDA completed successfully.")
    except Exception as e:
        logger.error(f"Error during EDA: {e}")
