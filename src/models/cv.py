import numpy as np
from sklearn.model_selection import BaseCrossValidator


class LeaveOneDayOutCV(BaseCrossValidator):
    """Leave-One-Day-Out cross validation based on day_of_week column."""

    def __init__(self, day_col="day_of_week"):
        self.day_col = day_col

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(np.unique(X[self.day_col]))

    def split(self, X, y=None, groups=None):
        days = np.asarray(X[self.day_col])
        unique_days = np.unique(days)
        for d in unique_days:
            val_idx = np.where(days == d)[0]
            trn_idx = np.where(days != d)[0]
            yield trn_idx, val_idx
