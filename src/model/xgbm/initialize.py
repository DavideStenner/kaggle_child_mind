import os
import json
import pickle
import logging

import pandas as pd
import polars as pl
import xgboost as xgb

from itertools import chain, product
from typing import Any, Union, Dict, Tuple
from src.base.model.initialize import ModelInit
from src.utils.logging_utils import get_logger


class XgbInit(ModelInit):
    def __init__(self, 
            experiment_name:str, 
            params_xgb: dict[str, Any],
            config_dict: dict[str, Any],
            fold_name: str = 'fold_info'
        ):
        
        self.inference: bool = False
        self.config_dict: dict[str, Any] = config_dict
        
        self.model_used: list[str] = ['main']
        self.model_metric_used: list[str] = {
            target_: {'label': 'q_kappa', 'maximize': True}
            for target_ in self.model_used
        }
        self.experiment_path: str = os.path.join(
            config_dict['PATH_EXPERIMENT'],
            experiment_name,
        )
        self.experiment_model_path: str = os.path.join(
            self.experiment_path, '{model_type}'
        )
        
        self.experiment_path_dict: dict[str, str] = {
            'feature_importance': os.path.join(
                self.experiment_model_path, 'insight'
            ),
            'insight': os.path.join(
                self.experiment_model_path, 'insight'
            ),
            'training': os.path.join(
                self.experiment_model_path, 'training'
            ),
            'shap': os.path.join(
                self.experiment_model_path, 'shap'
            ),
            'model': os.path.join(
                self.experiment_model_path, 'model'
            )
        }

        self.n_fold: int = config_dict['N_FOLD']
        
        self.fold_name: str = fold_name

        self.params_xgb: dict[str, Any] = params_xgb
        
        self.feature_list: list[str] = []
        
        self.get_categorical_columns()
        self.initialize_model_utils()
        self.get_model_file_name_dict()
        self.get_col_list()
        
    def get_col_list(self) -> None:

        self.useless_col_list: list[str] = (
            self.config_dict['COLUMN_INFO']['SPECIAL_COLUMNS'] +
            self.config_dict['COLUMN_INFO']['ORIGINAL_TARGET_LIST'] +
            [self.id_row] +
            [self.config_dict['COLUMN_INFO']['TARGET']] +
            [
                'fold_info', 'current_fold'
            ]
        )
        self.target_col: str = self.config_dict['COLUMN_INFO']['TARGET']
        

    def initialize_logger(self) -> None:
        self.training_logger: logging.Logger = get_logger(
            file_name='training_xgb.txt', path_logger=self.experiment_path
        )
        
    def initialize_model_utils(self) -> None:
        self.id_row: str = self.config_dict['ID_COL']
        
        self.feature_precision: str = 'float64'
        self.target_precision: str = 'float64'
        
        for model_type in self.model_used:
            setattr(
                self, f'model_{model_type}_list', [] 
            )
            
            setattr(
                self, f'progress_{model_type}_list', [] 
            )

                    
    def get_categorical_columns(self) -> None:
        #load all possible categorical feature
        self.categorical_col_list: list[str] = (
            self.config_dict['COLUMN_INFO']['STRING_FEATURE'] +
            self.config_dict['COLUMN_INFO']['CATEGORICAL_FEATURE']
        )
        
    def create_experiment_structure(self) -> None:
        if not os.path.isdir(self.experiment_path):
            os.makedirs(self.experiment_path)
            
        for model_type in self.model_used:
            if not os.path.isdir(self.experiment_model_path.format(model_type=model_type)):
                os.makedirs(self.experiment_model_path.format(model_type=model_type))
            
            for dir_path_format in self.experiment_path_dict.values():
                dir_path: str = dir_path_format.format(model_type=model_type)
                if not os.path.isdir(dir_path):
                    os.makedirs(dir_path)

    def load_model(self, target: str) -> None: 
        self.load_used_feature(target=target)
        self.load_used_categorical_feature(target=target)
        self.load_best_result(target=target)
        self.load_params()
        
        self.load_pickle_model_list(model_type=target)
    
    def get_model_file_name_dict(self) -> None:
        self.model_file_name_dict: dict[str, str] =  {
            'progress_list': {
                model_type: f'progress_{model_type}_list.pkl'
                for model_type in self.model_used
            },
            'best_result': {
                model_type: f'best_result_{model_type}_xgb.txt'
                for model_type in self.model_used
            },
            'model_pickle_list': {
                model_type: f'model_{model_type}_list_xgb.pkl'
                for model_type in self.model_used
            },
            'model_list': {
                model_type: f'xgb_{model_type}' + '_{fold_}.json'
                for model_type in self.model_used
            }
        }
    
    def save_progress_list(self, progress_list: list, model_type: str) -> None:
        with open(
            os.path.join(
                self.experiment_model_path.format(model_type=model_type),
                self.model_file_name_dict['progress_list'][model_type]
            ), 'wb'
        ) as file:
            pickle.dump(progress_list, file)

    def load_progress_list(self, model_type: str) -> list:
        with open(
            os.path.join(
                self.experiment_model_path.format(model_type=model_type),
                self.model_file_name_dict['progress_list'][model_type]
            ), 'rb'
        ) as file:
            progress_list = pickle.load(file)
            
        return progress_list
    def save_params(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'params_xgb.json'
            ), 'w'
        ) as file:
            json.dump(self.params_xgb, file)
    
    def load_params(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'params_xgb.json'
            ), 'r'
        ) as file:
            self.params_xgb = json.load(file)
    
    def save_best_result(self, best_result: dict[str, Union[int, float]], model_type: str) -> None:
        with open(
            os.path.join(
                self.experiment_model_path.format(model_type=model_type),
                self.model_file_name_dict['best_result'][model_type]
            ), 'w'
        ) as file:
            json.dump(best_result, file)
        
    def load_best_result(self, model_type: str) -> dict[str, Union[int, float]]:
        with open(
            os.path.join(
                self.experiment_model_path.format(model_type=model_type),
                self.model_file_name_dict['best_result'][model_type]
            ), 'r'
        ) as file:
            best_result = json.load(file)
            
        return best_result

    def save_pickle_model_list(self, model_list: list[xgb.Booster], model_type: str) -> None:
        with open(
            os.path.join(
                self.experiment_model_path.format(model_type=model_type),
                'model',
                self.model_file_name_dict['model_pickle_list'][model_type]
            ), 'wb'
        ) as file:
            pickle.dump(model_list, file)
    
    def load_pickle_model_list(self, model_type: str) -> list[xgb.Booster]:
        with open(
            os.path.join(
                self.experiment_path_dict['model'].format(model_type=model_type),
                self.model_file_name_dict['model_pickle_list'][model_type]
            ), 'rb'
        ) as file:
            model_list = pickle.load(file)
    
        return model_list

    def load_model_list(self, model_type: str, file_name: str) -> list[xgb.Booster]:
        
        return [
            xgb.Booster(
                params=self.params_xgb,
                model_file=os.path.join(
                    self.experiment_path_dict['model'].format(model_type=model_type),
                    file_name.format(fold_=fold_)
                )
            )
            for fold_ in range(self.n_fold)
        ]    
            
    def save_used_feature(self, model_type: str, feature_list: list[str]) -> None:
        with open(
            os.path.join(
                self.experiment_model_path.format(model_type=model_type),
                'used_feature.txt'
            ), 'w'
        ) as file:
            json.dump(
                {
                    'feature_model': feature_list
                }, 
                file
            )
    
    def load_used_categorical_feature(self, model_type: str) -> None:
        with open(
            os.path.join(
                self.experiment_model_path.format(model_type=model_type),
                'used_categorical_feature.txt'
            ), 'r'
        ) as file:
            self.categorical_col_list = json.load(file)['categorical_feature']
            
    def save_used_categorical_feature(self, model_type: str) -> None:
        with open(
            os.path.join(
                self.experiment_model_path.format(model_type=model_type),
                'used_categorical_feature.txt'
            ), 'w'
        ) as file:
            json.dump(
                {
                    'categorical_feature': self.categorical_col_list
                }, 
                file
            )

    def load_used_feature(self, model_type: str) -> None:
        with open(
            os.path.join(
                self.experiment_model_path.format(model_type=model_type),
                'used_feature.txt'
            ), 'r'
        ) as file:
            self.feature_list = json.load(file)['feature_model']