import numpy as np
import lightgbm as lgb
import xgboost as xgb

from typing import Tuple
from sklearn.metrics import cohen_kappa_score
# from pytorch_tabnet.metrics import Metric

from src.utils.import_utils import import_config

CONFIG: dict[str, any] = import_config()

def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def quadratic_weighted_kappa_tresh(combination: list[float], y_true: np.ndarray, y_pred: np.ndarray):
    rounded_prediciton_ = (
        np.where(
            y_pred < combination[0], 0,
            np.where(
                y_pred < combination[1], 1,
                    np.where(
                        y_pred < combination[2], 2, 
                        3
                    )
                )
        )
    )
    
    return quadratic_weighted_kappa(y_true, rounded_prediciton_)

# class QKappaTabnet(Metric):
#     def __init__(self):
#         self._name = "q_kappa"
#         self._maximize = True

#     def __call__(self, y_true, y_score):
#         # it run metric also on test for detector. needed to round for pseudo-labeling
#         y_true = y_true.round().astype(int)
#         y_score = y_score.round().astype(int)

#         eval_result = quadratic_weighted_kappa_tresh([0.5, 1.4, 2], y_true=y_true, y_pred=y_score)
#         return eval_result

class QuadraticKappa():         
    def is_max_optimal(self):
        return True

    def evaluate_round(self, approxes, target, weight):
        #it run metric also on test for detector. needed to round for pseudo-labeling
        y_true = np.array(target).round().astype(int)
        y_pred = approxes[0].round().astype(int)

        eval_result = quadratic_weighted_kappa(y_true=y_true, y_pred=y_pred)
        return eval_result
    
    def evaluate_tresh(self, approxes, target, weight):
        #it run metric also on test for detector. needed to round for pseudo-labeling
        y_true = np.array(target).round().astype(int)
        y_pred = approxes[0]

        eval_result = quadratic_weighted_kappa_tresh(combination=CONFIG['TRESHOLD'], y_true=y_true, y_pred=y_pred)
        return eval_result

    def evaluate(self, approxes, target, weight):
        eval_result = self.evaluate_tresh(approxes, target, weight)
        return eval_result, 1
    
    def get_final_error(self, error, weight):
        return error

def lgb_quadratic_kappa(y_pred: np.ndarray, data: lgb.Dataset) -> Tuple[str, float, bool]:
    rounded_y_pred = y_pred.round().astype(int)
    y_true = data.get_label()

    return 'q_kappa', quadratic_weighted_kappa(y_true=y_true, y_pred=rounded_y_pred), True

def lgb_quadratic_kappa_tresh(combination: list[float], y_pred: np.ndarray, data: lgb.Dataset) -> Tuple[str, float, bool]:
    y_true = data.get_label()

    return 'q_kappa', quadratic_weighted_kappa_tresh(combination=combination, y_true=y_true, y_pred=y_pred), True

def xgb_quadratic_kappa_tresh(combination: list[float], y_pred: np.ndarray, eval_data: xgb.DMatrix) -> Tuple[str, float]:
    y_true = eval_data.get_label()
    
    return 'q_kappa', quadratic_weighted_kappa_tresh(combination=combination, y_true=y_true, y_pred=y_pred)

def xgb_quadratic_kappa(y_pred: np.ndarray, eval_data: xgb.DMatrix) -> Tuple[str, float]:
    y_true = eval_data.get_label()
    rounded_y_pred = y_pred.round().astype(int)
    
    return 'q_kappa', quadratic_weighted_kappa(y_true=y_true, y_pred=rounded_y_pred)

def xgb_ordinal_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.cumsum(axis=1).argmax(axis=1)
    y_pred = (y_pred>0.5).cumsum(axis=1).argmax(axis=1)
    return quadratic_weighted_kappa(y_true=y_true, y_pred=y_pred)

def xgb_ordinal_quadratic_kappa(y_pred: np.ndarray, eval_data: xgb.DMatrix) -> Tuple[str, float]:
    y_true = eval_data.get_label().reshape(y_pred.shape)

    kappa_score = xgb_ordinal_kappa(y_true=y_true, y_pred=y_pred)
    return 'q_kappa', kappa_score