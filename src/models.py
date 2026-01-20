from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def get_baseline_model():
    """Simple baseline model"""
    return LogisticRegression(max_iter=1000, n_jobs=-1)

def get_random_forest(class_weight=None):
    """
    RandomForest for chemoinformatics.
    If class_weight='balanced', it handles imbalanced classes.
    """
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight=class_weight
    )
