import os
import json
import pickle
import logging

import numpy as np

from typing import Any, Union
from itertools import combinations
from src.base.model.initialize import ModelInit
from src.utils.logging_utils import get_logger
from src.model.nn.model import Standalone_RealMLP_TD_S_Regressor

class MLPInit(ModelInit):
    def __init__(self, 
            experiment_name:str, 
            params_nn: dict[str, Any],
            config_dict: dict[str, Any],
            fold_name: str = 'fold_info'
        ):
        
        self.inference: bool = False
        self.config_dict: dict[str, Any] = config_dict
        
        self.model_used: list[str] = ['main']
        self.model_metric_used: list[str] = {
            target_: {'label': 'l2', 'maximize': False}
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
            ),
            'model_param': os.path.join(
                self.experiment_model_path, 'model_param'
            ),
            'pseudo_labeling': os.path.join(
                self.experiment_model_path, 'pseudo_labeling'
            )
        }

        self.n_fold: int = config_dict['N_FOLD']
        
        self.fold_name: str = fold_name

        self.params_nn: dict[str, Any] = params_nn
        
        self.feature_list: list[str] = []
        self.original_path_gold: str = self.config_dict['PATH_GOLD_DATA']

        self.get_categorical_columns()
        self.initialize_model_utils()
        self.get_model_file_name_dict()
        self.get_col_list()

    
    @property
    def name(self) -> str:
        return 'mlp'
     
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
        
        self.feature_precision: str = 'float32'
        self.target_precision: str = 'float32'
        
        for model_type in self.model_used:
            setattr(
                self, f'model_{model_type}_list', [] 
            )
            setattr(
                self, f'model_param_{model_type}_list', [] 
            )
            setattr(
                self, f'progress_{model_type}_list', [] 
            )

    def set_postprocess_utils(self) -> None:
        max_range: int = (
            self.config_dict['COLUMN_INFO']['TARGET_N_UNIQUE'] if
            self.config_dict['COLUMN_INFO']['TARGET'] == 'sii'
            else 80
        )
        total_grid = np.linspace(0, max_range-1, 50)

        self.list_treshold_value: list[list[float]] = list(
            combinations(total_grid, 3)
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
                model_type: 'progress_list.pkl'
                for model_type in self.model_used
            },
            'best_result': {
                model_type: 'best_result_nn.txt'
                for model_type in self.model_used
            },
            'model_pickle_list': {
                model_type: 'model_list_nn.pkl'
                for model_type in self.model_used
            },
            'model_param_pickle_list': {
                model_type: 'model_list_nn.pkl'
                for model_type in self.model_used
            }
        }

    def load_best_pseudo_result(self) -> dict[str, any]:
        with open(
            os.path.join(
                self.experiment_path,
                'best_pseudo_result.json'
            ), 'r'
        ) as file:
            best_result = json.load(file)
            
        return best_result

    def save_best_pseudo_result(self, pseudo_result: dict[str, any]) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'best_pseudo_result.json'
            ), 'w'
        ) as file:
            json.dump(pseudo_result, file)

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
                'params_nn.json'
            ), 'w'
        ) as file:
            json.dump(self.params_nn, file)
    
    def load_params(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'params_nn.json'
            ), 'r'
        ) as file:
            self.params_nn = json.load(file)
    
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

    def save_pickle_model_param_list(self, model_param_list: list[any], model_type: str) -> None:
        with open(
            os.path.join(
                self.experiment_path_dict['model_param'].format(model_type=model_type),
                self.model_file_name_dict['model_param_pickle_list'][model_type]
            ), 'wb'
        ) as file:
            pickle.dump(model_param_list, file)

    def clear_pickle_model_param_list(self, model_type: str) -> None:
        file_path: str = (
            os.path.join(
                self.experiment_path_dict['model_param'].format(model_type=model_type),
                self.model_file_name_dict['model_param_pickle_list'][model_type]
            )
        )
        if os.path.isfile(file_path):
            os.remove(file_path)
            self.training_logger.info('Removed param list pickled!')
            
    def load_pickle_model_param_list(self, model_type: str) -> list[any]:
        with open(
            os.path.join(
                self.experiment_path_dict['model_param'].format(model_type=model_type),
                self.model_file_name_dict['model_pickle_list'][model_type]
            ), 'rb'
        ) as file:
            model_param_list = pickle.load(file)
    
        return model_param_list
    
    def save_pickle_model_list(self, model_list: list[Standalone_RealMLP_TD_S_Regressor], model_type: str) -> None:
        with open(
            os.path.join(
                self.experiment_path_dict['model'].format(model_type=model_type),
                self.model_file_name_dict['model_pickle_list'][model_type]
            ), 'wb'
        ) as file:
            pickle.dump(model_list, file)
    
    def load_pickle_model_list(self, model_type: str) -> list[Standalone_RealMLP_TD_S_Regressor]:
        with open(
            os.path.join(
                self.experiment_path_dict['model'].format(model_type=model_type),
                self.model_file_name_dict['model_pickle_list'][model_type]
            ), 'rb'
        ) as file:
            model_list = pickle.load(file)
    
        return model_list

            
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