import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Existing functions
# -----------------------------
def compute_roc_auc(y_true, y_pred_prob):
    return roc_auc_score(y_true, y_pred_prob)

def plot_roc_curve(y_true, y_pred_prob, title, save_path):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_true, y_pred_prob):.3f}")
    plt.plot([0,1],[0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()

# -----------------------------
# New Stage 6 functions
# -----------------------------
def compute_classification_metrics(y_true, y_pred):
    """
    Compute precision, recall, and F1-score for binary predictions.
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return precision, recall, f1

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """
    Plot and save confusion matrix using seaborn heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.savefig(save_path, dpi=300)
    plt.close()
def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """
    Plot top_n feature importances from a tree-based model (Random Forest, etc.)
    
    Parameters
    ----------
    model : fitted model (RandomForestClassifier, GradientBoosting, etc.)
    feature_names : list of str
        Names of features in the same order as model input
    top_n : int
        Number of top features to display
    save_path : str or Path
        File path to save the plot (optional)
    """
    # Get feature importances
    importances = model.feature_importances_
    
    # Create a DataFrame for easier sorting
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="skyblue")
    plt.xlabel("Feature Importance")
    plt.title("Top {} Feature Importances".format(top_n))
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()