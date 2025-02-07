import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Evaluating Recommendation Performance ---

def precision_at_k(actual, predicted, k=5):
    """
    Precision@K: Measures how many of the recommended items were actually purchased.
    """
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    
    if not predicted_set:
        return 0.0
    
    return len(actual_set & predicted_set) / len(predicted_set)

def recall_at_k(actual, predicted, k=5):
    """
    Recall@K: Measures how many of the purchased items were recommended.
    """
    actual_set = set(actual)
    predicted_set = set(predicted[:k])

    if not actual_set:
        return 0.0

    return len(actual_set & predicted_set) / len(actual_set)

def hit_rate(actual, predicted):
    """
    Hit Rate: Checks if any of the actual purchases are in the recommendations.
    """
    return int(bool(set(actual) & set(predicted)))

# --- 2. Visualizing Recommendation Performance ---

def plot_precision_recall_curve(actual_list, predicted_list, k_values):
    """
    Plots Precision vs Recall for different values of K.
    """
    precisions = []
    recalls = []

    for k in k_values:
        precision_scores = [precision_at_k(actual, predicted, k) for actual, predicted in zip(actual_list, predicted_list)]
        recall_scores = [recall_at_k(actual, predicted, k) for actual, predicted in zip(actual_list, predicted_list)]

        precisions.append(np.mean(precision_scores))
        recalls.append(np.mean(recall_scores))

    plt.figure(figsize=(8, 5))
    plt.plot(recalls, precisions, marker='o', linestyle='-', label='Precision-Recall Curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve for Recommendations")
    plt.legend()
    plt.grid(True)
    plt.show()
