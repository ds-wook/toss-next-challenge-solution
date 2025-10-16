import numpy as np
from sklearn.metrics import average_precision_score


# Metric functions
def calculate_weighted_logloss(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15
) -> float:
    """Calculate Weighted LogLoss with 50:50 class weights"""
    y_pred = np.clip(y_pred, eps, 1 - eps)

    mask_0 = y_true == 0
    mask_1 = y_true == 1

    # 1 - y_pred도 클리핑하여 log(0) 방지
    ll_0 = (
        -np.mean(np.log(np.clip(1 - y_pred[mask_0], eps, 1))) if mask_0.sum() > 0 else 0
    )
    ll_1 = -np.mean(np.log(np.clip(y_pred[mask_1], eps, 1))) if mask_1.sum() > 0 else 0

    return 0.5 * ll_0 + 0.5 * ll_1


def calculate_clipped_weighted_logloss(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15
) -> float:
    """Calculate Weighted LogLoss with 50:50 class weights (already clipped version)"""
    y_pred = np.clip(y_pred, eps, 1 - eps)

    mask_0 = y_true == 0
    mask_1 = y_true == 1

    # 1 - y_pred도 클리핑하여 log(0) 방지
    ll_0 = (
        -np.mean(np.log(np.clip(1 - y_pred[mask_0], eps, 1))) if mask_0.sum() > 0 else 0
    )
    ll_1 = -np.mean(np.log(np.clip(y_pred[mask_1], eps, 1))) if mask_1.sum() > 0 else 0

    return 0.5 * ll_0 + 0.5 * ll_1


def calculate_competition_score(
    y_true: np.ndarray, y_pred: np.ndarray, clip: bool = False
) -> tuple[float, float, float]:
    """Calculate competition score: 0.5*AP + 0.5*(1/(1+WLL))"""
    ap = average_precision_score(y_true, y_pred)
    if clip:
        wll = calculate_clipped_weighted_logloss(y_true, y_pred)
    else:
        wll = calculate_weighted_logloss(y_true, y_pred)
    score = 0.5 * ap + 0.5 * (1 / (1 + wll))
    return score, ap, wll


class CompetitionScoreMetric:
    def is_max_optimal(self):
        return True  # 점수가 높을수록 좋음

    def get_final_error(self, error, weight=None):
        # evaluate에서 반환한 error 배열의 평균
        return np.mean(error)

    def evaluate(self, approxes, target, weight=None):
        """
        approxes: list of [n_samples] predicted probabilities for class 1
        target: list of [n_samples] true labels
        weight: list or None
        """
        preds = 1 / (1 + np.exp(-np.array(approxes[0])))  # convert raw logits → prob
        y_true = np.array(target, dtype=int)

        score, _, _ = calculate_competition_score(y_true, preds)

        # CatBoost expects: (errors: np.ndarray, weights: np.ndarray)
        return np.array([score]), np.array([1.0])
