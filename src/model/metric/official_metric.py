import numpy as np
import lightgbm as lgb
import xgboost as xgb

from typing import Tuple
from sklearn.metrics import cohen_kappa_score


def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def lgb_quadratic_kappa(y_pred: np.ndarray, data: lgb.Dataset) -> Tuple[str, float, bool]:
    rounded_y_pred = y_pred.round().astype(int)
    y_true = data.get_label()

    return 'q_kappa', quadratic_weighted_kappa(y_true=y_true, y_pred=rounded_y_pred), True

def xgb_quadratic_kappa(y_pred: np.ndarray, eval_data: xgb.DMatrix) -> Tuple[str, float]:
    y_true = eval_data.get_label().reshape(y_pred.shape).cumsum(axis=1).argmax(axis=1)
    y_pred = (y_pred>0.5).cumsum(axis=1).argmax(axis=1)

    return 'q_kappa', quadratic_weighted_kappa(y_true=y_true, y_pred=y_pred)