import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Configure Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def preprocess_data(df):
    """
    Prepares data for training the purchase prediction model.
    - Selects relevant features.
    - Handles missing values.
    - Splits into train-test sets.
    - Scales numerical features.
    """
    logger.info("Preprocessing data for purchase prediction...")

    # Define target variable: 1 if purchase event exists, else 0
    df['purchase'] = (df['session_purchase_count'] > 0).astype(int)

    # Define feature columns
    feature_cols = [
        'session_duration', 'session_view_count', 'session_cart_count', 
        'product_view_count', 'product_purchase_count', 'user_event_count', 'price_sensitivity'
    ]

    # Drop rows with missing values in selected features
    df = df.dropna(subset=feature_cols)

    X = df[feature_cols]
    y = df['purchase']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train, model_type='random_forest'):
    """
    Trains a classification model to predict purchase likelihood.
    - Supports 'random_forest' and 'xgboost'.
    """
    logger.info(f"Training {model_type} model...")

    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'xgboost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError("Invalid model_type. Choose 'random_forest' or 'xgboost'.")

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on test data and logs key performance metrics.
    """
    logger.info("Evaluating model performance...")
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logger.info(f"Model Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    print("Classification Report:\n", classification_report(y_test, y_pred))

    return accuracy, precision, recall, f1

def train_and_evaluate(df, model_type='random_forest'):
    """
    Full pipeline: preprocess data, train model, evaluate performance.
    """
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    model = train_model(X_train, y_train, model_type)
    metrics = evaluate_model(model, X_test, y_test)

    return model, scaler, metrics

# if __name__ == "__main__":
#     df_features = pd.read_csv("processed_with_features.csv")
#     train_and_evaluate(df_features, model_type='random_forest')
