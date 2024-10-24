import numpy as np
import lightgbm as lgb

from typing import Tuple
from sklearn.metrics import cohen_kappa_score


def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def lgb_quadratic_kappa(y_pred: np.ndarray, data: lgb.Dataset) -> Tuple[str, float, bool]:
    rounded_y_pred = y_pred.round().astype(int)
    y_true = data.get_label()

    return 'q_kappa', quadratic_weighted_kappa(y_true=y_true, y_pred=rounded_y_pred), True
