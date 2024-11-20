import os
import json
import pickle
import logging

import numpy as np
import polars as pl
import lightgbm as lgb

from typing import Any, Union, Dict, Tuple
from itertools import combinations
from src.base.ensemble.initialize import EnsembleInit
from src.utils.logging_utils import get_logger

from src.model.lgbm.pipeline import LgbmPipeline
from src.model.xgbm.pipeline import XgbPipeline
from src.model.ctb.pipeline import CtbPipeline

class EnsembleInit(EnsembleInit):
    def __init__(self, 
            experiment_name:str, 
            params_ensemble: dict[str, Any],
            config_dict: dict[str, Any],
        ):
        self.inference: bool = False
        self.config_dict: dict[str, Any] = config_dict
            
        self.experiment_path: str = os.path.join(
            config_dict['PATH_EXPERIMENT'],
            experiment_name,
        )
        self.n_fold: int = config_dict['N_FOLD']
        
        self.params_ensemble: dict[str, Any] = params_ensemble
        
        self.all_feature_list: list[str] = []
        self.original_path_gold: str = self.config_dict['PATH_GOLD_DATA']
        self.target_col: str = self.config_dict['COLUMN_INFO']['TARGET']
        
        self.initialize_model_utils()
        self.create_experiment_structure()
        
    def create_experiment_structure(self) -> None:
        if not os.path.isdir(self.experiment_path):
            os.makedirs(self.experiment_path)

    def initialize_logger(self) -> None:
        self.ensemble_logger: logging.Logger = get_logger(
            file_name='ensemble_log.txt', path_logger=self.experiment_path
        )
        
    def initialize_model_utils(self) -> None:
        self.id_row: str = self.config_dict['ID_COL']

        self.feature_precision: str = 'float64'
        self.target_precision: str = 'float64'

        self.pipeline_model_list: list[Union[LgbmPipeline, XgbPipeline, CtbPipeline]] = []
        
    def set_postprocess_utils(self) -> None:
        total_grid = np.arange(0, self.config_dict['COLUMN_INFO']['TARGET_N_UNIQUE']-1, step=0.025)

        self.list_treshold_value: list[list[float]] = list(
            combinations(total_grid, 3)
        )
                                    
    def save_params(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'params_ensemble.json'
            ), 'w'
        ) as file:
            json.dump(self.params_ensemble, file)
    
    def load_params(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'params_ensemble.json'
            ), 'r'
        ) as file:
            self.params_ensemble = json.load(file)
            
    def save_best_result(self, best_result: dict[str, Union[int, float]]) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'best_result.json'
            ), 'w'
        ) as file:
            json.dump(best_result, file)
    
    def load_best_result(self) -> dict[str, Union[int, float]]:
        with open(
            os.path.join(
                self.experiment_path,
                'best_result.json'
            ), 'r'
        ) as file:
            best_result = json.load(file)
        
        return best_result
    
    def load_model(self) -> None:
        if 'lgb' in self.params_ensemble['model']:
            from src.utils.import_utils import import_params
            from src.model.lgbm.pipeline import LgbmPipeline
        
            params_model, experiment_name = import_params(model='lgb')
        
            lgb_model: LgbmPipeline = LgbmPipeline(
                experiment_name=experiment_name + "_lgb",
                params_lgb=params_model,
                config_dict=self.config_dict,
                evaluate_shap=False,
            )
            self.pipeline_model_list.append(lgb_model)
            
            
        if 'xgb' in self.params_ensemble['model']:
            from src.utils.import_utils import import_params
            from src.model.xgbm.pipeline import XgbPipeline
        
            params_model, experiment_name = import_params(model='xgb')
        
            xgb_model: XgbPipeline = XgbPipeline(
                experiment_name=experiment_name + "_xgb",
                params_xgb=params_model,
                config_dict=self.config_dict,
                evaluate_shap=False,
            )

            self.pipeline_model_list.append(xgb_model)

        if 'ctb' in self.params_ensemble['model']:
            from src.utils.import_utils import import_params
            from src.model.ctb.pipeline import CtbPipeline
        
            params_model, experiment_name = import_params(model='ctb')
        
            ctb_model: CtbPipeline = CtbPipeline(
                experiment_name=experiment_name + "_ctb",
                params_ctb=params_model,
                config_dict=self.config_dict,
                evaluate_shap=False,
            )

            self.pipeline_model_list.append(ctb_model)
