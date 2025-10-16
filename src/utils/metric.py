from typing import List
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
)


def compute_metrics(y_true: List[int], y_pred_proba: List[float]):
    # Calculate AUC score
    auc_score = roc_auc_score(y_true, y_pred_proba)

    # Generate classification report (convert probabilities to binary predictions)
    y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred_proba]

    macro_f1 = f1_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary, pos_label=1)
    precision = precision_score(y_true, y_pred_binary, pos_label=1)

    return {
        "auc": auc_score,
        "macro_f1": macro_f1,
        "recall": recall,
        "precision": precision,
    }
